# teste_completo.py (VERSÃO COM LÓGICA AVANÇADA DE UNIFICAÇÃO E DEBUG)

import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
from collections import defaultdict

import torch
import onnxruntime as ort
from torchvision.ops import nms

# ==============================================================================
# CONFIGURAÇÕES DO TESTE
# ==============================================================================
IMAGE_FOLDER_PATH = "/mnt/arch/home/servidor/output_20m_teste_clahe2/"
MODELS = {
    'trincas': 'trincas.onnx',
    'panelas': 'panelas.onnx' 
}
CLASSES = {
    'trincas': ['fc3', 'courodejacare'],
    'panelas': ['panela', 'remendo']
}
# Pasta para salvar imagens de debug das máscaras
DEBUG_MASKS_FOLDER = "debug_masks"

print("--- INICIANDO SCRIPT DE TESTE AVANÇADO (COM MORFOLOGIA E DEBUG) ---")

# ==============================================================================
# CLASSE DE INFERÊNCIA ONNX (Inalterada)
# ==============================================================================
class YOLOSegmentationONNX:
    def __init__(self, model_path: str, classes: List[str], input_size: tuple = (1024, 1024),
                 conf_threshold: float = 0.4, nms_threshold: float = 0.5):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.classes = classes
        self.num_classes = len(classes)
        self.session = self._create_inference_session(model_path)
        self.output_names = [output.name for output in self.session.get_outputs()]

    def _create_inference_session(self, model_path: str) -> ort.InferenceSession:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_path}")
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
# LÓGICA DE PROCESSAMENTO (AGORA COM MORFOLOGIA)
# ==============================================================================
def unify_masks_and_get_polygons(detections: List[Dict], h: int, w: int, offset_x: int, offset_y: int,
                                 image_name: str, analysis_type: str, slice_index: int) -> List[Dict]:
    """
    Unifica as máscaras por classe, aplica operações morfológicas, encontra
    contornos e salva imagens de debug.
    """
    if not detections:
        return []

    debug_folder = Path(DEBUG_MASKS_FOLDER)
    masks_by_class, scores_by_class = defaultdict(list), defaultdict(list)
    for det in detections:
        masks_by_class[det['class_name']].append(det['mask'])
        scores_by_class[det['class_name']].append(det['score'])

    final_polygons_info = []
    for class_name, masks in masks_by_class.items():
        # 1. Criar a máscara unificada ("layer")
        unified_mask = np.zeros((h, w), dtype=np.uint8)
        for mask in masks:
            if mask.shape[0] == h and mask.shape[1] == w:
                unified_mask = np.maximum(unified_mask, mask * 255) # Usamos 255 para visualização

        # 2. Salvar máscara intermediária ANTES da morfologia
        debug_filename_before = f"{image_name}_{analysis_type}_slice{slice_index}_{class_name}_before.png"
        cv2.imwrite(str(debug_folder / debug_filename_before), unified_mask)

        # 3. Aplicar operação morfológica de FECHAMENTO
        # Um kernel maior conecta objetos mais distantes. 5x5 é um bom começo.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_mask = cv2.morphologyEx(unified_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 4. Salvar máscara intermediária DEPOIS da morfologia
        debug_filename_after = f"{image_name}_{analysis_type}_slice{slice_index}_{class_name}_after.png"
        cv2.imwrite(str(debug_folder / debug_filename_after), closed_mask)

        # 5. Encontrar contornos na máscara tratada
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 50: continue
            
            # Usamos um epsilon bem pequeno, já que a máscara foi tratada
            epsilon = 1.0 
            approx_poly = cv2.approxPolyDP(contour, epsilon, True)
            
            x, y, cw, ch = cv2.boundingRect(contour)
            global_polygon = approx_poly.squeeze(axis=1) + [offset_x, offset_y]
            
            final_polygons_info.append({
                'class_name': class_name, 'avg_score': np.mean(scores_by_class[class_name]),
                'global_polygon': global_polygon.tolist(),
                'global_box': [x + offset_x, y + offset_y, x + cw + offset_x, y + ch + offset_y],
                'area': int(cv2.contourArea(contour))
            })
    return final_polygons_info

def run_trincas_analysis(model: YOLOSegmentationONNX, large_image: np.ndarray, image_name: str) -> List[Dict]:
    print("\n--- INICIANDO ANÁLISE DE TRINCAS ---")
    final_results = []
    slice_size = 1024
    img_h, img_w, _ = large_image.shape
    slice_idx_counter = 0
    for y in tqdm(range(0, img_h, slice_size), desc="Análise de Trincas"):
        for x in range(0, img_w, slice_size):
            slice_np = large_image[y:y + slice_size, x:x + slice_size]
            if slice_np.size == 0: continue
            slice_detections = model.predict(slice_np)
            h, w, _ = slice_np.shape
            unified_polygons = unify_masks_and_get_polygons(slice_detections, h, w, x, y, image_name, 'trincas', slice_idx_counter)
            for poly_info in unified_polygons:
                poly_info['analise'] = 'trincas'
                final_results.append(poly_info)
            slice_idx_counter += 1
    return final_results

def run_panelas_analysis(model: YOLOSegmentationONNX, large_image: np.ndarray, image_name: str) -> List[Dict]:
    print("\n--- INICIANDO ANÁLISE DE PANELAS/REMENDOS ---")
    final_results = []
    h, w, _ = large_image.shape
    num_strips = 5
    strip_width = w // num_strips
    for i in tqdm(range(num_strips), desc="Análise de Panelas"):
        offset_x = i * strip_width
        slice_np = large_image[:, offset_x : offset_x + strip_width]
        if slice_np.size == 0: continue
        slice_detections = model.predict(slice_np)
        slice_h, slice_w, _ = slice_np.shape
        unified_polygons = unify_masks_and_get_polygons(slice_detections, slice_h, slice_w, offset_x, 0, image_name, 'panelas', i)
        for poly_info in unified_polygons:
            poly_info['analise'] = 'panelas'
            final_results.append(poly_info)
    return final_results

# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    try:
        # Cria a pasta de debug
        debug_folder = Path(DEBUG_MASKS_FOLDER)
        debug_folder.mkdir(exist_ok=True)
        print(f"Máscaras de debug serão salvas em: '{debug_folder.resolve()}'")
        
        # 1. Carregar modelos
        print("Carregando modelos ONNX...")
        model_trincas = YOLOSegmentationONNX(model_path=MODELS['trincas'], classes=CLASSES['trincas'])
        model_panelas = YOLOSegmentationONNX(model_path=MODELS['panelas'], classes=CLASSES['panelas'])
        print("Modelos carregados com sucesso.")

        # 2. Encontrar imagens na pasta
        image_folder = Path(IMAGE_FOLDER_PATH)
        image_files = list(image_folder.glob('*.png'))
        if not image_files:
            raise FileNotFoundError(f"Nenhuma imagem .png encontrada em {IMAGE_FOLDER_PATH}")
        print(f"\nEncontradas {len(image_files)} imagens para processar.")

        # 3. Loop de processamento
        all_results = {}
        for image_path in tqdm(image_files, desc="Processando Arquivos", unit="arquivo"):
            image_name_stem = image_path.stem
            print(f"\n--- Processando: {image_path.name} ---")
            large_image = cv2.imread(str(image_path))
            if large_image is None:
                print(f"Aviso: Falha ao carregar a imagem {image_path.name}, pulando.")
                continue
            
            current_image_detections = []
            current_image_detections.extend(run_trincas_analysis(model_trincas, large_image, image_name_stem))
            current_image_detections.extend(run_panelas_analysis(model_panelas, large_image, image_name_stem))
            all_results[image_path.name] = current_image_detections

        # 4. Salvar resultados finais
        print("\n\n--- ANÁLISE COMPLETA CONCLUÍDA ---")
        output_path = "debug_results_lote_final.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print(f"\nResultados completos salvos em '{output_path}'")

    except Exception as e:
        print(f"\n--- OCORREU UM ERRO FATAL ---")
        import traceback
        traceback.print_exc()