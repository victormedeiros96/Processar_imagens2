# Core/processing_script.py

import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Generator, Tuple

# --- Novas importações para o ONNX ---
import torch
import onnxruntime as ort
from torchvision.ops import nms

# --- LISTA DE CLASSES (AJUSTE PARA O SEU MODELO) ---
MY_CLASSES = ['fc3', 'courodejacare']

# ==============================================================================
# CLASSE DE INFERÊNCIA ONNX
# ==============================================================================
class YOLOSegmentationONNX:
    """
    Classe para realizar inferência com um modelo de SEGMENTAÇÃO YOLO exportado como ONNX.
    """
    def __init__(self, model_path: str, classes: List[str],
                 input_size: tuple = (1024, 1024),
                 conf_threshold: float = 0.4, nms_threshold: float = 0.5):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.classes = classes
        self.num_classes = len(classes)
        self.session = self._create_inference_session(model_path)
        model_outputs = self.session.get_outputs()
        self.output_names = [output.name for output in model_outputs]
        self.output_shape_det = model_outputs[0].shape
        self.num_mask_coeffs = self.output_shape_det[2] - self.num_classes - 4

    def _create_inference_session(self, model_path: str) -> ort.InferenceSession:
        try:
            sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            print(f"ONNX: Usando dispositivo: {sess.get_providers()[0]}")
            return sess
        except Exception as e:
            raise RuntimeError(f"Erro ao criar sessão ONNX: {e}")

    def _preprocess(self, image: np.ndarray) -> (np.ndarray, float, tuple):
        original_h, original_w = image.shape[:2]
        scale = min(self.input_size[0] / original_h, self.input_size[1] / original_w)
        new_h, new_w = int(original_h * scale), int(original_w * scale)
        resized_image = cv2.resize(image, (new_w, new_h))
        padded_image = np.full((self.input_size[0], self.input_size[1], 3), 114, dtype=np.uint8)
        padded_image[:new_h, :new_w] = resized_image
        image_tensor = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_tensor = np.transpose(image_tensor, (2, 0, 1))
        return np.expand_dims(image_tensor, axis=0), scale, (original_h, original_w)

    def _postprocess(self, outputs: List[np.ndarray], scale: float, orig_shape: tuple) -> List[Dict]:
        predictions = outputs[0][0]
        mask_prototypes = outputs[1][0]
        boxes_raw, scores, class_ids, mask_coeffs = predictions[:, :4], np.max(predictions[:, 4:4+self.num_classes], axis=1), np.argmax(predictions[:, 4:4+self.num_classes], axis=1), predictions[:, 4+self.num_classes:]
        mask_conf = scores > self.conf_threshold
        if not np.any(mask_conf): return []
        boxes, scores, class_ids, mask_coeffs = boxes_raw[mask_conf], scores[mask_conf], class_ids[mask_conf], mask_coeffs[mask_conf]
        x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxes_torch = torch.from_numpy(np.stack([x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2], axis=1))
        indices = nms(boxes_torch, torch.from_numpy(scores), self.nms_threshold)
        final_detections = []
        for i in indices:
            mask = self.process_mask(mask_prototypes, mask_coeffs[i], boxes_torch[i], orig_shape, scale)
            final_detections.append({
                'box': (boxes_torch[i].numpy() / scale).astype(int), 'score': scores[i].item(),
                'class_id': class_ids[i].item(), 'class_name': self.classes[class_ids[i].item()], 'mask': mask
            })
        return final_detections
        
    def process_mask(self, prototypes: np.ndarray, mask_coeffs: np.ndarray, box: torch.Tensor, orig_shape: tuple, scale: float) -> np.ndarray:
        c, mh, mw = prototypes.shape
        mask = (mask_coeffs.reshape(1, -1) @ prototypes.reshape(c, -1)).reshape(mh, mw)
        mask = 1 / (1 + np.exp(-mask))
        orig_h, orig_w = orig_shape
        x1, y1, x2, y2 = (box.numpy() / scale).astype(int)
        mask_resized = cv2.resize(mask, (orig_w, orig_h))
        final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        y1, y2 = max(0, y1), min(orig_h, y2)
        x1, x2 = max(0, x1), min(orig_w, x2)
        final_mask[y1:y2, x1:x2] = mask_resized[y1:y2, x1:x2] > 0.5
        return final_mask

    def predict(self, image: np.ndarray) -> List[Dict]:
        input_tensor, scale, orig_shape = self._preprocess(image)
        outputs = self.session.run(self.output_names, {self.session.get_inputs()[0].name: input_tensor})
        return self._postprocess(outputs, scale, orig_shape)

# ==============================================================================
# CLASSE ORQUESTRADORA (LÓGICA DE FATIAMENTO)
# ==============================================================================
class AIImageProcessor:
    """
    Mantém a lógica de fatiamento, mas usa o YOLOSegmentationONNX para inferência.
    """
    def __init__(self, model_path: str, input_folder: str, output_folder: str,
                 slice_height: int = 1024,
                 slice_width: int = 1024):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_json_path = self.output_folder / 'analysis_results_onnx.json'
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.model = YOLOSegmentationONNX(
            model_path=model_path,
            classes=MY_CLASSES,
            input_size=(self.slice_width, self.slice_height)
        )

    def _create_slices(self, image: np.ndarray) -> Generator[Tuple[np.ndarray, Tuple[int, int]], None, None]:
        img_h, img_w, _ = image.shape
        for y in range(0, img_h, self.slice_height):
            for x in range(0, img_w, self.slice_width):
                yield image[y:y + self.slice_height, x:x + self.slice_width], (y, x)

    def run_analysis(self):
        image_files = list(self.input_folder.glob('*.[jp][pn]g')) + list(self.input_folder.glob('*.jpeg'))
        if not image_files:
            print("Nenhuma imagem encontrada.")
            return

        all_results = {}
        for img_path in tqdm(image_files, desc="Processando Imagens Grandes", unit="img"):
            try:
                large_image = cv2.imread(str(img_path))
                if large_image is None: continue

                detections_for_this_image = []
                slices_generator = self._create_slices(large_image)

                for slice_np, (offset_y, offset_x) in slices_generator:
                    slice_detections = self.model.predict(slice_np)
                    for det in slice_detections:
                        box = det['box']
                        # --- CORREÇÃO APLICADA AQUI ---
                        # Converte explicitamente as coordenadas e o score para tipos nativos do Python.
                        global_box = [
                            int(box[0] + offset_x), 
                            int(box[1] + offset_y), 
                            int(box[2] + offset_x), 
                            int(box[3] + offset_y)
                        ]
                        score = float(det['score'])
                        
                        detections_for_this_image.append({
                            'class_name': det['class_name'],
                            'score': score,
                            'global_box': global_box,
                            'mask_area_on_slice': int(np.sum(det['mask']))
                        })
                all_results[img_path.name] = detections_for_this_image
            except Exception as e:
                print(f"\nErro ao processar a imagem {img_path.name}: {e}")
        self._save_to_json(all_results)

    def _save_to_json(self, data: Dict):
        self.output_folder.mkdir(parents=True, exist_ok=True)
        with open(self.output_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\nResultados salvos com sucesso em: {self.output_json_path}")
