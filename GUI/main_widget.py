# GUI/main_widget.py

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog, QHBoxLayout, QCheckBox

class MainWidget(QWidget):
    start_requested = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Processador de Imagens")

        # Layout principal
        layout = QVBoxLayout(self)

        # Seleção de pasta de entrada
        layout.addWidget(QLabel("Selecione a pasta com as imagens de entrada:"))
        self.input_folder_edit = QLineEdit('/mnt/arch/home/servidor/output_20m_teste_clahe2/')
        self.input_folder_button = QPushButton("Procurar...")
        self.input_folder_button.clicked.connect(self.select_input_folder)
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_folder_edit)
        input_layout.addWidget(self.input_folder_button)
        layout.addLayout(input_layout)

        # Seleção de pasta de saída
        layout.addWidget(QLabel("Selecione a pasta para salvar os resultados:"))
        self.output_folder_edit = QLineEdit('/home/servidor/Deletar_VICTOR_Teste_trinca/teste')
        self.output_folder_button = QPushButton("Procurar...")
        self.output_folder_button.clicked.connect(self.select_output_folder)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_folder_edit)
        output_layout.addWidget(self.output_folder_button)
        layout.addLayout(output_layout)

        # --- NOVO: Checkboxes para seleção de análise ---
        layout.addWidget(QLabel("Selecione os tipos de análise:"))
        self.cb_trincas = QCheckBox("Analisar Trincas")
        self.cb_trincas.setChecked(True)
        self.cb_panelas = QCheckBox("Analisar Panelas e Remendos")
        self.cb_panelas.setChecked(True)
        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(self.cb_trincas)
        checkbox_layout.addWidget(self.cb_panelas)
        layout.addLayout(checkbox_layout)

        # Botão de iniciar
        self.start_button = QPushButton("Iniciar Processamento")
        self.start_button.clicked.connect(self.on_start)
        layout.addWidget(self.start_button)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Entrada")
        if folder:
            self.input_folder_edit.setText(folder)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Saída")
        if folder:
            self.output_folder_edit.setText(folder)

    def on_start(self):
        input_folder = self.input_folder_edit.text()
        output_folder = self.output_folder_edit.text()
        if input_folder and output_folder:
            self.start_requested.emit(input_folder, output_folder)
        else:
            # Pode-se adicionar uma QMessageBox de aviso aqui
            print("Por favor, selecione as pastas de entrada e saída.")

    def toggle_controls(self, is_enabled: bool):
        self.input_folder_button.setEnabled(is_enabled)
        self.output_folder_button.setEnabled(is_enabled)
        self.start_button.setEnabled(is_enabled)
        self.cb_trincas.setEnabled(is_enabled)
        self.cb_panelas.setEnabled(is_enabled)