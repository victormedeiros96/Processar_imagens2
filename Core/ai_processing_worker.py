# Core/ai_processing_worker.py

from PyQt6.QtCore import QObject, pyqtSignal
from Core.ai_processing_script import AIImageProcessor

class AIProcessingWorker(QObject):
    """
    Worker que executa o processamento de IA em uma thread separada.
    """
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, input_folder: str, output_folder: str):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.is_running = True

    def run(self):
        """
        O método que será executado pela QThread.
        """
        try:
            # 1. Defina o caminho para o seu modelo treinado
            model_path = 'best.pt' # IMPORTANTE: Coloque o caminho correto aqui!
            
            # 2. Instancia o processador de IA
            # Você pode ajustar os parâmetros como 'device', 'batch_size', etc.
            processor = AIImageProcessor(
                model_path=model_path,
                output_folder=self.output_folder,
                device='0', # '0' para GPU 0, 'cpu' para CPU
                batch_size=8
            )
            
            # 3. Executa a análise
            if self.is_running:
                processor.run_analysis(self.input_folder)

        except Exception as e:
            # Emite um sinal de erro se algo der errado
            self.error.emit(f"Ocorreu um erro: {e}")
        finally:
            # Garante que o sinal 'finished' seja emitido no final
            if self.is_running:
                self.finished.emit()

    def stop(self):
        """Permite que a thread principal solicite a parada do worker."""
        print("Solicitando a parada do worker...")
        self.is_running = False