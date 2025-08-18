# GUI/main_window.py

from PyQt6.QtWidgets import QMainWindow, QMessageBox
from PyQt6.QtCore import QThread, pyqtSlot

from GUI.main_widget import MainWidget
from Core.processing_worker import ProcessingWorker # <-- MUDANÇA AQUI

class MainWindow(QMainWindow):
    """
    Orquestrador que gerencia a execução do processamento em uma thread.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analisador de Imagens com IA")
        self.resize(500, 300)

        self._thread = None
        self._worker = None

        self.main_widget = MainWidget()
        self.setCentralWidget(self.main_widget)
        
        self.main_widget.start_requested.connect(self.handle_start_request)

    @pyqtSlot(str, str)
    def handle_start_request(self, input_folder, output_folder):
        """Prepara e inicia o processamento de IA em uma nova thread."""
        if self._thread and self._thread.isRunning():
            QMessageBox.warning(self, "Aviso", "Um processo já está em execução.")
            return
            
        self.main_widget.toggle_controls(False)
        self.statusBar().showMessage("Iniciando processamento de IA em segundo plano...")

        self._thread = QThread()
        # Instancia o novo worker de IA
        self._worker = ProcessingWorker(input_folder, output_folder) # <-- MUDANÇA AQUI
        self._worker.moveToThread(self._thread)

        # Conecta os sinais do worker aos slots da janela
        self._worker.finished.connect(self._on_process_finished)
        self._worker.error.connect(self._on_process_error)
        self._thread.started.connect(self._worker.run)
        
        # Limpeza da thread quando ela termina
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._worker.error.connect(self._thread.quit) # Também para a thread em caso de erro

        self._thread.start()

    def _on_process_finished(self):
        """Chamado quando o worker termina com sucesso."""
        self.statusBar().showMessage("Processo finalizado com sucesso!", 5000)
        self.main_widget.toggle_controls(True)
        QMessageBox.information(self, "Sucesso", "A análise das imagens foi concluída.")
        
    def _on_process_error(self, error_message):
        """Chamado se o worker emitir um erro."""
        QMessageBox.critical(self, "Erro no Processamento", error_message)
        self.main_widget.toggle_controls(True)

    def closeEvent(self, event):
        """Garante que a thread seja encerrada ao fechar a janela."""
        if self._thread and self._thread.isRunning():
            self._worker.stop()
            self._thread.quit()
            self._thread.wait(5000) # Espera até 5s pela thread
        event.accept()