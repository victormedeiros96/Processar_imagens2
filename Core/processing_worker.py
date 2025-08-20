# Core/processing_worker.py

from PyQt6.QtCore import QObject, pyqtSignal
from typing import List
from Core.processing_script import AIImageProcessor

class ProcessingWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, input_folder: str, output_folder: str, analyses_to_run: List[str]):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.analyses_to_run = analyses_to_run
        self.is_running = True

    def run(self):
        try:
            model_paths = {
                'trincas': 'trincas.onnx',
                'panelas': 'panelas.onnx'
            }
            
            if self.is_running:
                processor = AIImageProcessor(
                    models=model_paths,
                    input_folder=self.input_folder,
                    output_folder=self.output_folder,
                    analyses_to_run=self.analyses_to_run
                )
                processor.run_analysis()

        except Exception as e:
            self.error.emit(f"Ocorreu um erro durante o processamento: {e}")
        finally:
            if self.is_running:
                self.finished.emit()

    def stop(self):
        self.is_running = False