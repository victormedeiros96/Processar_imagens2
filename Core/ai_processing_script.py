# Core/ai_processing_script.py

import cv2
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple, Generator, Optional
from tqdm import tqdm # <--- ADICIONADO AQUI

# Nota: A função 'cv2.ximgproc.thinning' requer a instalação do pacote opencv-contrib.
# Instale com: pip install opencv-contrib-python

def apply_clahe_grayscale(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    if img.ndim == 3:
        gray_channel = img[:, :, 0]
    else:
        gray_channel = img

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_channel = clahe.apply(gray_channel)
    img_clahe_3_channel = cv2.merge([clahe_channel, clahe_channel, clahe_channel])
    return img_clahe_3_channel

class AIImageProcessor:
    """
    Classe principal para encapsular a lógica de análise de imagem com IA.
    Esta classe será compilada com Cython.
    """
    def __init__(self,
                 model_path: str,
                 output_folder: str,
                 device: str = 'cpu',
                 batch_size: int = 1,
                 slice_height: int = 1024,
                 slice_width: int = 1024):
        """Inicializa o processador de IA."""
        self.output_path = Path(output_folder) / 'analysis_results.json'
        self.device = device
        self.batch_size = batch_size
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.model = self._load_model(model_path)
        print(f"Modelo YOLO carregado. Device: '{self.device}'.")

    def _load_model(self, model_path: str) -> YOLO:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_path}")
        return YOLO(model_path)

    def _create_slices_from_image(self, large_image: np.ndarray) -> Generator[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]], None, None]:
        img_h, img_w, _ = large_image.shape
        for r_idx, y in enumerate(range(0, img_h, self.slice_height)):
            for c_idx, x in enumerate(range(0, img_w, self.slice_width)):
                yield large_image[y:y + self.slice_height, x:x + self.slice_width], (r_idx, c_idx), (y, x)

    @staticmethod
    def _simplify_mask_to_polygon(mask_np: np.ndarray, tolerance: float = 1.5) -> List[List[int]]:
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return []
        main_contour = max(contours, key=cv2.contourArea)
        simplified_contour = cv2.approxPolyDP(main_contour, tolerance, True)
        return simplified_contour.squeeze().tolist() if simplified_contour.size > 0 else []

    @staticmethod
    def _calculate_direction(mask_tensor: np.ndarray, image_shape: tuple, box_coords: list) -> Optional[str]:
        h, w = image_shape
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = (255 * mask_tensor.cpu().numpy()).clip(0, 255).astype(np.uint8)
        x1, y1, x2, y2 = map(int, box_coords)
        subimage = mask[y1:y2, x1:x2]
        if subimage.size == 0: return None
        thinned = cv2.ximgproc.thinning(cv2.morphologyEx(subimage, cv2.MORPH_OPEN, kernel))
        ys, xs = np.nonzero(thinned)
        if len(xs) > 1 and len(ys) > 1:
            return 'Horizontal' if (np.max(xs) - np.min(xs)) > (np.max(ys) - np.min(ys)) else 'Vertical'
        return None

    def _process_single_result(self, result: Any, slice_index: Tuple[int, int], slice_origin: Tuple[int, int]) -> List[Dict[str, Any]]:
        output_objects = []
        if not (result.boxes and result.masks): return output_objects
        offset_y, offset_x = slice_origin
        for box, mask_tensor, cls_tensor in zip(result.boxes, result.masks.data, result.boxes.cls):
            class_id, local_bbox = int(cls_tensor.item()), box.xyxy[0].tolist()
            global_bbox = [local_bbox[0] + offset_x, local_bbox[1] + offset_y, local_bbox[2] + offset_x, local_bbox[3] + offset_y]
            full_mask_np = (255 * mask_tensor.cpu().numpy()).clip(0, 255).astype(np.uint8)
            local_polygon = self._simplify_mask_to_polygon(full_mask_np, 1.2)
            detection_info = {
                "class": class_id,
                "slice_index": f"({slice_index[0]}, {slice_index[1]})",
                "global_bbox": global_bbox,
                "global_polygon": [[pt[0] + offset_x, pt[1] + offset_y] for pt in local_polygon],
                "area": int(np.count_nonzero(full_mask_np)) if class_id == 1 else None,
                "direction": self._calculate_direction(mask_tensor, result.orig_shape, local_bbox) if class_id == 0 else None
            }
            output_objects.append(detection_info)
        return output_objects

    def run_analysis(self, image_dir: str):
        """
        Método principal que executa a análise em um diretório de imagens.
        Este método será chamado pelo worker.
        """
        image_path = Path(image_dir)
        if not image_path.is_dir():
            raise FileNotFoundError(f"O diretório de entrada não foi encontrado: {image_dir}")

        image_files = list(image_path.glob('*.[jp][pn]g')) + list(image_path.glob('*.jpeg'))
        if not image_files:
            print("Nenhuma imagem encontrada no diretório.")
            return

        all_results = {}
        # --- MUDANÇA AQUI: Adicionado tqdm para a barra de progresso ---
        for img_file in tqdm(image_files, desc="Analisando Imagens com IA", unit="img"):
            try:
                large_image_np = cv2.imread(str(img_file))
                if large_image_np is None: continue

                large_image_clahe = apply_clahe_grayscale(large_image_np)
                slices_generator = self._create_slices_from_image(large_image_clahe)
                all_slices_meta = list(slices_generator)
                detections_for_this_image = []

                for i in range(0, len(all_slices_meta), self.batch_size):
                    batch_meta = all_slices_meta[i:i + self.batch_size]
                    batch_slices_np = [meta[0] for meta in batch_meta]
                    if not batch_slices_np: continue

                    results_list = self.model.predict(batch_slices_np, verbose=False, device=self.device)

                    for result, meta in zip(results_list, batch_meta):
                        detections_for_this_image.extend(self._process_single_result(result, meta[1], meta[2]))

                all_results[img_file.name] = detections_for_this_image
            except Exception as e:
                print(f"Erro ao processar a imagem {img_file.name}: {e}")

        self._save_to_json(all_results)

    def _save_to_json(self, data: Dict):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Resultados da análise salvos em: {self.output_path}")