# GUI/main_window.py

from PyQt6.QtWidgets import QMainWindow, QMessageBox
from PyQt6.QtCore import QThread, pyqtSlot
from GUI.main_widget import MainWidget
from Core.processing_worker import ProcessingWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analisador de Imagens")
        self.resize(500, 250)
        self._thread = None
        self._worker = None
        self.main_widget = MainWidget()
        self.setCentralWidget(self.main_widget)
        self.main_widget.start_requested.connect(self.handle_start_request)

    @pyqtSlot(str, str)
    def handle_start_request(self, input_folder, output_folder):
        if self._thread and self._thread.isRunning():
            QMessageBox.warning(self, "Aviso", "Um processo já está em execução.")
            return

        # --- NOVO: Lê os checkboxes antes de iniciar ---
        analyses_to_run = []
        if self.main_widget.cb_trincas.isChecked():
            analyses_to_run.append('trincas')
        if self.main_widget.cb_panelas.isChecked():
            analyses_to_run.append('panelas')
            
        if not analyses_to_run:
            QMessageBox.warning(self, "Aviso", "Selecione pelo menos um tipo de análise.")
            self.main_widget.toggle_controls(True)
            return

        self.main_widget.toggle_controls(False)
        self.statusBar().showMessage("Iniciando processamento...")
        self._thread = QThread()
        self._worker = ProcessingWorker(input_folder, output_folder, analyses_to_run)
        self._worker.moveToThread(self._thread)
        self._worker.finished.connect(self._on_process_finished)
        self._worker.error.connect(self._on_process_error)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_process_finished(self):
        self.statusBar().showMessage("Processo finalizado com sucesso!", 5000)
        self.main_widget.toggle_controls(True)
        QMessageBox.information(self, "Sucesso", "A análise das imagens foi concluída.")
        
    def _on_process_error(self, error_message):
        QMessageBox.critical(self, "Erro no Processamento", error_message)
        self.main_widget.toggle_controls(True)

    def closeEvent(self, event):
        if self._thread and self._thread.isRunning():
            self._worker.stop()
            self._thread.quit()
            self._thread.wait(5000)
        event.accept()