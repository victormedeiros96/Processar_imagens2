# visualizar_resultados.py

import cv2
import json
import numpy as np
from pathlib import Path

# --- CONFIGURAÇÕES ---
# 1. Aponte para a pasta com as imagens originais
PASTA_IMAGENS_ORIGINAIS = Path("/mnt/arch/home/servidor/output_20m_teste_clahe2/")

# 2. Aponte para o arquivo JSON gerado pela análise
ARQUIVO_JSON_RESULTADOS = Path("/home/servidor/Deletar_VICTOR_Teste_trinca/teste/analise_completa.json")

# 3. Aponte para a pasta onde as imagens com as máscaras serão salvas
PASTA_SAIDA_VISUALIZACAO = Path("/home/servidor/Deletar_VICTOR_Teste_trinca/visualizacao_resultados/")
# --- FIM DAS CONFIGURAÇÕES ---

def plotar_mascaras(imagem_original: np.ndarray, deteccoes: list) -> np.ndarray:
    """
    Desenha os polígonos de detecção em uma imagem.
    
    Args:
        imagem_original: A imagem (carregada com OpenCV) onde desenhar.
        deteccoes: A lista de dicionários de detecção para esta imagem, vinda do JSON.
        
    Returns:
        Uma nova imagem com as máscaras desenhadas.
    """
    # Cria uma cópia para não modificar a original
    overlay = imagem_original.copy()

    for defeito in deteccoes:
        # Pega o polígono do JSON. Precisa converter para um array numpy.
        poligono = np.array(defeito['global_polygon'], dtype=np.int32)
        classe = defeito['class_name']
        analise = defeito['analise']

        # Define uma cor com base no tipo de análise
        # (BGR para o OpenCV)
        cor = (0, 0, 255) # Vermelho para trincas
        if analise == 'panelas':
            cor = (255, 0, 0) # Azul para panelas

        # Desenha o polígono preenchido na imagem de overlay
        cv2.fillPoly(overlay, [poligono], cor)
        
        # Desenha o contorno do polígono na imagem original
        cv2.polylines(imagem_original, [poligono], isClosed=True, color=cor, thickness=2)

    # Mistura a imagem original com a camada de overlay semi-transparente
    imagem_final = cv2.addWeighted(overlay, 0.4, imagem_original, 0.6, 0)
    
    return imagem_final


if __name__ == "__main__":
    # Cria a pasta de saída se ela não existir
    PASTA_SAIDA_VISUALIZACAO.mkdir(exist_ok=True)
    print(f"Resultados serão salvos em: {PASTA_SAIDA_VISUALIZACAO.resolve()}")

    # Carrega os dados do arquivo JSON
    try:
        with open(ARQUIVO_JSON_RESULTADOS, 'r', encoding='utf-8') as f:
            resultados = json.load(f)
        print(f"Arquivo JSON '{ARQUIVO_JSON_RESULTADOS.name}' carregado com sucesso.")
    except FileNotFoundError:
        print(f"ERRO: Arquivo JSON não encontrado em '{ARQUIVO_JSON_RESULTADOS}'")
        exit()
    except json.JSONDecodeError:
        print(f"ERRO: O arquivo JSON '{ARQUIVO_JSON_RESULTADOS.name}' está mal formatado.")
        exit()

    # Itera sobre cada imagem que tem resultados no JSON
    for nome_imagem, deteccoes in resultados.items():
        if not deteccoes:
            print(f"Aviso: Nenhuma detecção para a imagem '{nome_imagem}', pulando.")
            continue

        caminho_imagem_original = PASTA_IMAGENS_ORIGINAIS / nome_imagem
        
        # Verifica se a imagem original existe
        if not caminho_imagem_original.exists():
            print(f"ERRO: Imagem original '{nome_imagem}' não encontrada em '{PASTA_IMAGENS_ORIGINAIS}'")
            continue

        print(f"Processando '{nome_imagem}'...")
        # Carrega a imagem original
        imagem_original = cv2.imread(str(caminho_imagem_original))
        
        if imagem_original is None:
            print(f"ERRO: Falha ao carregar a imagem '{caminho_imagem_original}' com o OpenCV.")
            continue
            
        # Chama a função para desenhar as máscaras
        imagem_com_mascaras = plotar_mascaras(imagem_original, deteccoes)
        
        # Salva o resultado
        nome_arquivo_saida = f"resultado_{Path(nome_imagem).stem}.jpg"
        caminho_saida = PASTA_SAIDA_VISUALIZACAO / nome_arquivo_saida
        
        # Salva a imagem com alta qualidade
        cv2.imwrite(str(caminho_saida), imagem_com_mascaras, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f" -> Imagem com máscaras salva como '{nome_arquivo_saida}'")

    print("\nVisualização de todos os resultados concluída.")