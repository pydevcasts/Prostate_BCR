import os
import sys
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QWidget, QFileDialog, QTextEdit, QMainWindow, QLabel, QPushButton, QLineEdit, QVBoxLayout


app = QApplication(sys.argv)
window = QWidget()
# ======================
window.setWindowTitle("Basic Widget ...")
window.resize(300,200)
window.setStyleSheet('background:skyblue')
laber_user = QLabel("Username: ")
input_user = QLineEdit()
input_user.setStyleSheet('padding:8px; font-size:16px; font-weight:bold;color:black;border:3px solid red;border-radius:20px')
label_pass = QLabel("Password: ")
input_pass = QLineEdit()
input_pass.setStyleSheet('padding:8px; font-size:16px; font-weight:bold;color:black;border:3px solid red;border-radius:20px')
input_pass.setEchoMode(QLineEdit.EchoMode.Password)

login_button = QPushButton("Login")
login_button.setStyleSheet("""
    QPushButton {
        background-color: red;
        font-size: 16px;
        margin-top: 10px;
        padding: 8px;
        border: 3px solid red;
        border-radius: 20px;
        color: white;
    }
    QPushButton:hover {
        color: black;
        background-color: yellow;
    }
""")
#Create vertical layout
layout = QVBoxLayout()
layout.addWidget(laber_user)
layout.addWidget(input_user)
layout.addWidget(label_pass)
layout.addWidget(input_pass)
layout.addWidget(login_button)

window.setLayout(layout)

# # =====================
window.show()
sys.exit(app.exec())




# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
    
#         self.setWindowTitle("Menu Example")
#         self.resize(500,400)

#         self.editor = QTextEdit()
#         self.setCentralWidget(self.editor)

#         menu_bar = self.menuBar()
#         file_menu = menu_bar.addMenu("File")

#         new_action = QAction("New", self)
#         open_action = QAction("open", self)
#         save_action = QAction("Save", self)
#         exit_action = QAction("Exit", self)

#         file_menu.addAction(new_action)
#         file_menu.addSeparator()
#         file_menu.addAction(open_action)
#         file_menu.addSeparator()
#         file_menu.addAction(save_action)
#         file_menu.addSeparator()
#         file_menu.addAction(exit_action)

#         exit_action.triggered.connect(self.close)
#         open_action.triggered.connect(self.open_file)
#         save_action.triggered.connect(self.save_file)
    
#     def save_file(self):
#         file_path, _ = QFileDialog.getSaveFileName(self, "Save File", '', 'Save file (*.txt)')
#         if file_path:
#             with open(file_path, "w", encoding='utf-8') as file:
#                 file.write(self.editor.toPlainText())
    
#     def open_file(self):
#         file_path, _ = QFileDialog.getOpenFileName(self, "Open File", '', 'Text file (*.txt)')
#         if file_path:
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 self.editor.setText(file.read())
    
                

# app = QApplication(sys.argv)
# window = MainWindow()
# window.show()
# sys.exit(app.exec())