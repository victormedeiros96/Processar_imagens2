import onnxruntime as ort
import numpy as np

MODEL_PATH = "best.onnx" # Certifique-se que o modelo está no mesmo diretório

try:
    print(f"Carregando o modelo: {MODEL_PATH}")
    session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # Pega as informações de entrada do modelo
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape
    print(f"Nome da entrada: {input_name}")
    print(f"Shape da entrada esperado: {input_shape}")

    # Cria uma imagem falsa (tensor com ruído) com o shape correto
    # Se o shape for dinâmico (com 'batch' ou 'height'), usamos valores padrão
    batch_size = 1
    channels = 3
    height = 640  # Usando 640 como padrão
    width = 640   # Usando 640 como padrão
    
    # Ajusta para o formato NCHW (Batch, Canais, Altura, Largura)
    dummy_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)

    print(f"Shape da entrada que estamos enviando: {dummy_input.shape}")

    # Executa a inferência
    outputs = session.run(None, {input_name: dummy_input})
    
    # --- A PARTE MAIS IMPORTANTE ---
    # Imprime o shape do tensor de saída
    output_tensor = outputs[0]
    print("\n--- RESULTADO DO DIAGNÓSTICO ---")
    print(f"O SHAPE DO TENSOR DE SAÍDA É: {output_tensor.shape}")
    print("---------------------------------")


except Exception as e:
    print(f"Ocorreu um erro: {e}")