import pyttsx3
import sys
from PyQt5.QtWidgets import QApplication, QTextEdit
from PyQt5.QtCore import Qt
import pygame
pygame.mixer.init()
import threading

engine = pyttsx3.init()
engine.setProperty('rate', 100)
class PronounceApp(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Text to speech')
        self.setPlaceholderText("Type here and press Enter to listen...")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            text = self.toPlainText().strip()
            if text:
                engine.say(text)
                threading.Thread(target=lambda : engine.runAndWait()).start()
        else:
            super().keyPressEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PronounceApp()
    ex.show()
    sys.exit(app.exec_())
