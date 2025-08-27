# Core/processing_script.py (VERSÃO COM LÓGICA DE UNIFICAÇÃO GLOBAL)

import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from collections import defaultdict

import torch
import onnxruntime as ort
from torchvision.ops import nms

# ==============================================================================
# CLASSE DE INFERÊNCIA ONNX (Inalterada)
# ==============================================================================
class YOLOSegmentationONNX:
    # ... (O código desta classe permanece exatamente o mesmo)
    def __init__(self, model_path: str, classes: List[str],
                 input_size: tuple = (1024, 1024),
                 conf_threshold: float = 0.4, nms_threshold: float = 0.5):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.classes = classes
        self.num_classes = len(classes)
        self.session = self._create_inference_session(model_path)
        self.output_names = [output.name for output in self.session.get_outputs()]

    def _create_inference_session(self, model_path: str) -> ort.InferenceSession:
        try:
            sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            print(f"ONNX: Usando dispositivo: {sess.get_providers()[0]} para o modelo {Path(model_path).name}")
            return sess
        except Exception as e:
            raise RuntimeError(f"Erro ao criar sessão ONNX para {model_path}: {e}")

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
        scores = np.max(predictions[:, 4:4+self.num_classes], axis=1)
        conf_mask = scores > self.conf_threshold
        if not np.any(conf_mask): return []
        predictions_filtered, scores_filtered = predictions[conf_mask], scores[conf_mask]
        boxes_data = predictions_filtered[:, :4]
        class_ids_data = np.argmax(predictions_filtered[:, 4:4+self.num_classes], axis=1)
        mask_coeffs_data = predictions_filtered[:, 4+self.num_classes:]
        x_c, y_c, w, h = boxes_data[:, 0], boxes_data[:, 1], boxes_data[:, 2], boxes_data[:, 3]
        boxes_torch = torch.from_numpy(np.stack([x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2], axis=1))
        scores_torch = torch.from_numpy(scores_filtered)
        indices = nms(boxes_torch, scores_torch, self.nms_threshold)
        final_detections = []
        for i in indices:
            mask = self.process_mask(mask_prototypes, mask_coeffs_data[i], boxes_torch[i], orig_shape, scale)
            final_detections.append({
                'score': scores_filtered[i].item(),
                'class_name': self.classes[class_ids_data[i].item()],
                'mask': mask
            })
        return final_detections

    def process_mask(self, prototypes: np.ndarray, mask_coeffs: np.ndarray, box: torch.Tensor, orig_shape: tuple, scale: float) -> np.ndarray:
        c, mh, mw = prototypes.shape
        if mask_coeffs.shape[0] != c:
             raise ValueError(f"Incompatibilidade de dimensões na máscara: coeffs={mask_coeffs.shape[0]}, protos={c}")
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
# CLASSE ORQUESTRADORA (LÓGICA REESCRITA PARA UNIFICAÇÃO GLOBAL)
# ==============================================================================
class AIImageProcessor:
    def __init__(self, models: Dict[str, str], input_folder: str, output_folder: str,
                 analyses_to_run: List[str]):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_json_path = self.output_folder / 'analise_completa.json'
        self.analyses_to_run = analyses_to_run
        
        self.debug_folder = self.output_folder / "debug_masks"
        self.debug_folder.mkdir(exist_ok=True)
        print(f"Máscaras de debug serão salvas em: '{self.debug_folder.resolve()}'")

        self.model_trincas = None
        if 'trincas' in self.analyses_to_run:
            self.model_trincas = YOLOSegmentationONNX(
                model_path=models['trincas'], classes=['fc3', 'courodejacare'], input_size=(1024, 1024)
            )

        self.model_panelas = None
        if 'panelas' in self.analyses_to_run:
            self.model_panelas = YOLOSegmentationONNX(
                model_path=models['panelas'], classes=['panela', 'remendo'], input_size=(4096, 4096)
            )
            
    def _extract_polygons_from_global_mask(self, global_mask_dict: Dict[str, np.ndarray], image_name: str, analysis_type: str) -> List[Dict]:
        """
        Aplica morfologia e extrai polígonos da máscara global final.
        """
        final_polygons_info = []
        for class_name, global_mask in global_mask_dict.items():
            
            # Salva a máscara global ANTES da morfologia para debug
            debug_filename_before = f"{image_name}_{analysis_type}_{class_name}_global_before.png"
            cv2.imwrite(str(self.debug_folder / debug_filename_before), global_mask)

            # Aplica operação morfológica de FECHAMENTO para unir defeitos
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) # Kernel pode ser ajustado
            closed_mask = cv2.morphologyEx(global_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

            # Salva a máscara global DEPOIS da morfologia
            debug_filename_after = f"{image_name}_{analysis_type}_{class_name}_global_after.png"
            cv2.imwrite(str(self.debug_folder / debug_filename_after), closed_mask)

            contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 100: continue # Aumentar área mínima para a máscara global
                
                epsilon = 1.5 
                approx_poly = cv2.approxPolyDP(contour, epsilon, True)
                
                # Coordenadas já são globais, não é necessário offset
                x, y, cw, ch = cv2.boundingRect(contour)
                
                final_polygons_info.append({
                    'class_name': class_name,
                    'global_polygon': approx_poly.squeeze(axis=1).tolist(),
                    'global_box': [x, y, x + cw, y + ch],
                    'area': int(cv2.contourArea(contour))
                    # 'avg_score' foi removido pois agregar scores de forma precisa seria complexo
                })
        return final_polygons_info

    def run_analysis(self):
        image_files = list(self.input_folder.glob('*.[jp][pn]g')) + list(self.input_folder.glob('*.jpeg'))
        if not image_files:
            print("Nenhuma imagem encontrada.")
            return

        all_results = {}
        for img_path in tqdm(image_files, desc="Processando Arquivos", unit="arquivo", position=0):
            image_name_stem = img_path.stem
            try:
                large_image = cv2.imread(str(img_path))
                if large_image is None: continue
                
                img_h, img_w, _ = large_image.shape
                detections_for_this_image = []

                # --- ANÁLISE DE TRINCAS ---
                if 'trincas' in self.analyses_to_run and self.model_trincas:
                    print("\nFase 1/2 (Trincas): Detectando e montando máscara global...")
                    # Cria máscaras globais em branco, uma para cada classe
                    trincas_global_masks = {cls: np.zeros((img_h, img_w), dtype=np.uint8) for cls in self.model_trincas.classes}
                    
                    slice_size = 1024
                    for y in tqdm(range(0, img_h, slice_size), desc="Detectando Trincas", leave=False, position=1):
                        for x in range(0, img_w, slice_size):
                            slice_np = large_image[y:y + slice_size, x:x + slice_size]
                            if slice_np.size == 0: continue
                            
                            slice_detections = self.model_trincas.predict(slice_np)
                            # Cola as máscaras detectadas na máscara global
                            for det in slice_detections:
                                h, w = det['mask'].shape
                                trincas_global_masks[det['class_name']][y:y+h, x:x+w] = np.maximum(
                                    trincas_global_masks[det['class_name']][y:y+h, x:x+w],
                                    det['mask'] * 255
                                )
                    
                    print("Fase 2/2 (Trincas): Extraindo polígonos da máscara global...")
                    trincas_polygons = self._extract_polygons_from_global_mask(trincas_global_masks, image_name_stem, 'trincas')
                    for poly_info in trincas_polygons:
                        poly_info['analise'] = 'trincas'
                    detections_for_this_image.extend(trincas_polygons)

                # --- ANÁLISE DE PANELAS ---
                if 'panelas' in self.analyses_to_run and self.model_panelas:
                    print("\nFase 1/2 (Panelas): Detectando e montando máscara global...")
                    panelas_global_masks = {cls: np.zeros((img_h, img_w), dtype=np.uint8) for cls in self.model_panelas.classes}

                    slice_size = 4096
                    for y in tqdm(range(0, img_h, slice_size), desc="Detectando Panelas", leave=False, position=1):
                        for x in range(0, img_w, slice_size):
                            slice_np = large_image[y:y + slice_size, x:x + slice_size]
                            if slice_np.size == 0: continue
                            
                            slice_detections = self.model_panelas.predict(slice_np)
                            for det in slice_detections:
                                h, w = det['mask'].shape
                                panelas_global_masks[det['class_name']][y:y+h, x:x+w] = np.maximum(
                                    panelas_global_masks[det['class_name']][y:y+h, x:x+w],
                                    det['mask'] * 255
                                )
                                
                    print("Fase 2/2 (Panelas): Extraindo polígonos da máscara global...")
                    panelas_polygons = self._extract_polygons_from_global_mask(panelas_global_masks, image_name_stem, 'panelas')
                    for poly_info in panelas_polygons:
                        poly_info['analise'] = 'panelas'
                    detections_for_this_image.extend(panelas_polygons)

                all_results[img_path.name] = detections_for_this_image
            except Exception as e:
                print(f"\nErro ao processar a imagem {img_path.name}: {e}")
                import traceback
                traceback.print_exc()

        self._save_to_json(all_results)

    def _save_to_json(self, data: Dict):
        self.output_folder.mkdir(parents=True, exist_ok=True)
        with open(self.output_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\nResultados salvos com sucesso em: {self.output_json_path}")