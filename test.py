import torch
import cv2
import numpy as np
import onnxruntime as ort
from torchvision.ops import nms
from typing import List, Dict

# --- LISTA DE CLASSES (AJUSTADA PARA O SEU MODELO) ---
MY_CLASSES = ['fc3', 'courodejacare']


class YOLOSegmentationONNX:
    """
    Classe para realizar inferência com um modelo de SEGMENTAÇÃO YOLO exportado como ONNX.
    """
    def __init__(self, model_path: str, classes: List[str], input_size: tuple = (640, 640), 
                 conf_threshold: float = 0.4, nms_threshold: float = 0.5):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.classes = classes
        self.num_classes = len(classes)
        
        self.session = self._create_inference_session(model_path)
        
        # O modelo de segmentação tem duas saídas
        model_outputs = self.session.get_outputs()
        self.output_names = [output.name for output in model_outputs]
        
        # A primeira saída contém as detecções, a segunda os protótipos de máscara
        self.output_shape_det = model_outputs[0].shape
        self.output_shape_proto = model_outputs[1].shape
        
        # Calcula o número de coeficientes da máscara a partir do shape da saída de detecção
        self.num_mask_coeffs = self.output_shape_det[1] - self.num_classes - 4

    def _create_inference_session(self, model_path: str) -> ort.InferenceSession:
        try:
            sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            if 'CUDAExecutionProvider' in sess.get_providers():
                print("Usando dispositivo: GPU (CUDA)")
            else:
                print("Usando dispositivo: CPU")
            return sess
        except Exception as e:
            print(f"Erro ao criar a sessão de inferência: {e}")
            raise

    def _preprocess(self, image: np.ndarray) -> (np.ndarray, float, tuple):
        original_h, original_w = image.shape[:2]
        scale = min(self.input_size[0] / original_h, self.input_size[1] / original_w)
        new_h, new_w = int(original_h * scale), int(original_w * scale)
        resized_image = cv2.resize(image, (new_w, new_h))
        padded_image = np.full((self.input_size[0], self.input_size[1], 3), 114, dtype=np.uint8)
        padded_image[:new_h, :new_w] = resized_image
        image_tensor = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        image_tensor = image_tensor.astype(np.float32) / 255.0
        image_tensor = np.transpose(image_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(image_tensor, axis=0)
        return input_tensor, scale, (original_h, original_w)

    def _postprocess(self, outputs: List[np.ndarray], scale: float, orig_shape: tuple) -> List[Dict]:
        detections_output = outputs[0][0]  # Shape: (38, 8400)
        mask_prototypes = outputs[1][0]    # Shape: (32, 160, 160)

        predictions = np.transpose(detections_output) # Shape: (8400, 38)
        
        # Separa as partes do vetor de predição
        boxes = predictions[:, :4]
        scores = np.max(predictions[:, 4:4+self.num_classes], axis=1)
        class_ids = np.argmax(predictions[:, 4:4+self.num_classes], axis=1)
        mask_coeffs = predictions[:, 4+self.num_classes:]

        # Filtra por confiança
        mask_conf = scores > self.conf_threshold
        boxes, scores, class_ids, mask_coeffs = boxes[mask_conf], scores[mask_conf], class_ids[mask_conf], mask_coeffs[mask_conf]

        if boxes.shape[0] == 0:
            return []

        # Converte caixas para (x1, y1, x2, y2) e aplica NMS
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        boxes_torch = torch.from_numpy(np.stack([x1, y1, x2, y2], axis=1))
        scores_torch = torch.from_numpy(scores)
        indices = nms(boxes_torch, scores_torch, self.nms_threshold)

        # Processa apenas os resultados que sobreviveram ao NMS
        final_detections = []
        for i in indices:
            # Multiplica os coeficientes da máscara pelos protótipos para gerar a máscara
            mask_weights = mask_coeffs[i]
            mask = self.process_mask(mask_prototypes, mask_weights, boxes_torch[i], orig_shape, scale)

            detection = {
                'box': (boxes_torch[i].numpy() / scale).tolist(),
                'score': scores[i].item(),
                'class_id': class_ids[i].item(),
                'class_name': self.classes[class_ids[i].item()],
                'mask': mask
            }
            final_detections.append(detection)

        return final_detections

    def process_mask(self, prototypes: np.ndarray, mask_coeffs: np.ndarray, box: torch.Tensor, orig_shape: tuple, scale: float) -> np.ndarray:
        # Multiplicação matricial para combinar protótipos e coeficientes
        c, mh, mw = prototypes.shape
        coeffs = mask_coeffs.reshape(1, -1) # Shape (1, 32)
        protos_reshaped = prototypes.reshape(c, -1) # Shape (32, 160*160)
        
        mask = (coeffs @ protos_reshaped).reshape(mh, mw)
        mask = 1 / (1 + np.exp(-mask)) # Função Sigmoid

        # Redimensiona a máscara e a recorta para o tamanho final
        orig_h, orig_w = orig_shape
        box_scaled = (box.numpy() / scale).astype(int)
        x1, y1, x2, y2 = box_scaled

        mask = cv2.resize(mask, (orig_w, orig_h))
        
        # Cria uma máscara binária final
        final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        final_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2] > 0.5
        
        return final_mask

    def predict(self, image_path: str) -> (np.ndarray, List[Dict]):
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"Não foi possível ler a imagem: {image_path}")
        
        input_tensor, scale, orig_shape = self._preprocess(original_image)
        
        # O modelo retorna uma lista com os dois tensores de saída
        outputs = self.session.run(self.output_names, {self.session.get_inputs()[0].name: input_tensor})
        
        detections = self._postprocess(outputs, scale, orig_shape)
        
        return original_image, detections
    
    @staticmethod
    def draw_results(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        overlay = image.copy()
        for i, det in enumerate(detections):
            box = np.array(det['box']).astype(int)
            score = det['score']
            class_name = det['class_name']
            mask = det['mask']

            # Gera uma cor aleatória para cada classe para visualização
            color = np.random.randint(0, 255, 3).tolist()
            
            # Desenha a máscara
            overlay[mask == 1] = color

            # Desenha a caixa delimitadora e o rótulo
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Combina a imagem original com a sobreposição da máscara
        return cv2.addWeighted(overlay, 0.5, image, 0.5, 0)


if __name__ == "__main__":
    MODEL_PATH = "best.onnx"
    IMAGE_PATH = "/mnt/ssd3/ufmt_test_right_4m/frame_1.png"
    
    try:
        detector = YOLOSegmentationONNX(model_path=MODEL_PATH, classes=MY_CLASSES)
        original_image, detections = detector.predict(IMAGE_PATH)
        
        print(f"\n{len(detections)} objetos segmentados:")
        for det in detections:
            print(f"  - Classe: {det['class_name']}, Confiança: {det['score']:.2f}")
        
        output_image = YOLOSegmentationONNX.draw_results(original_image.copy(), detections)
        
        output_filename = "resultado_segmentacao_onnx.jpg"
        cv2.imwrite(output_filename, output_image)
        print(f"\nImagem com a segmentação salva como '{output_filename}'")
        
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")