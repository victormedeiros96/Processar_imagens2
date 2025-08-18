# Core/processing_worker.py

from PyQt6.QtCore import QObject, pyqtSignal
from Core.processing_script import AIImageProcessor

class ProcessingWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, input_folder: str, output_folder: str):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.is_running = True

    def run(self):
        try:
            model_path = 'best.onnx'
            if self.is_running:
                processor = AIImageProcessor(
                    model_path=model_path,
                    input_folder=self.input_folder,
                    output_folder=self.output_folder
                )
                processor.run_analysis()
        except Exception as e:
            self.error.emit(f"Ocorreu um erro durante o processamento: {e}")
        finally:
            if self.is_running:
                self.finished.emit()

    def stop(self):
        self.is_running = False