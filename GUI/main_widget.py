# GUI/main_widget.py

from PyQt6 import QtWidgets as QtW
from PyQt6.QtCore import pyqtSignal

class MainWidget(QtW.QWidget):
    """Interface do usuário, agora sem a barra de progresso."""
    start_requested = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.setLayout(QtW.QVBoxLayout())
        
        # --- Widgets de Input ---
        self.input_label = QtW.QLabel("Pasta de Origem:")
        self.select_input_button = QtW.QPushButton("Selecionar Pasta de Origem")
        self.input_path_edit = QtW.QLineEdit()
        self.input_path_edit.setReadOnly(True)
        
        self.layout().addWidget(self.input_label)
        self.layout().addWidget(self.select_input_button)
        self.layout().addWidget(self.input_path_edit)
        
        # --- Widgets de Output ---
        self.output_label = QtW.QLabel("Pasta de Saída:")
        self.select_output_button = QtW.QPushButton("Selecionar Pasta de Saída")
        self.output_path_edit = QtW.QLineEdit()
        self.output_path_edit.setReadOnly(True)
        
        self.layout().addWidget(self.output_label)
        self.layout().addWidget(self.select_output_button)
        self.layout().addWidget(self.output_path_edit)
        
        # --- Separador e Ação ---
        separator = QtW.QFrame()
        separator.setFrameShape(QtW.QFrame.Shape.HLine)
        separator.setFrameShadow(QtW.QFrame.Shadow.Sunken)
        self.layout().addWidget(separator)

        self.start_button = QtW.QPushButton("▶ Iniciar Processo")
        self.start_button.setStyleSheet("font-size: 14px; padding: 8px;")
        self.layout().addWidget(self.start_button)
        
        # --- Conexões internas ---
        self.select_input_button.clicked.connect(self._open_input_directory_dialog)
        self.select_output_button.clicked.connect(self._open_output_directory_dialog)
        self.start_button.clicked.connect(self._on_start_button_clicked)
        
        self.layout().addStretch()

    def _open_input_directory_dialog(self):
        directory = QtW.QFileDialog.getExistingDirectory(self, "Selecione a pasta de origem")
        if directory:
            self.input_path_edit.setText(directory)
            
    def _open_output_directory_dialog(self):
        directory = QtW.QFileDialog.getExistingDirectory(self, "Selecione a pasta de saída")
        if directory:
            self.output_path_edit.setText(directory)

    def _on_start_button_clicked(self):
        """Valida os campos e emite o sinal para o orquestrador."""
        input_folder = self.input_path_edit.text()
        output_folder = self.output_path_edit.text()
        
        if not input_folder or not output_folder:
            QtW.QMessageBox.warning(self, "Erro", "Por favor, selecione a pasta de origem e a de saída.")
            return
        
        self.start_requested.emit(input_folder, output_folder)

    def toggle_controls(self, is_enabled: bool):
        """Habilita ou desabilita os controles de interação."""
        self.start_button.setEnabled(is_enabled)
        self.select_input_button.setEnabled(is_enabled)
        self.select_output_button.setEnabled(is_enabled)