import sys
from PyQt5.QtWidgets import QApplication, QTextEdit
from PyQt5.QtCore import Qt, QThread
from gtts import gTTS
#from playsound import playsound
import pygame
pygame.mixer.init()
import sounddevice as sd
import numpy as np
import threading

class EchoWorker(QThread):
    def __init__(self):
        super().__init__()
        self.samplerate = int(sd.query_devices(None, 'input')['default_samplerate'])
        self.delay_buffer = np.zeros(int(self.samplerate * 0.5), dtype=np.float32)

    def callback(self, indata, outdata, frames, time, status):
        outdata[:] = self.delay_buffer[:frames].reshape(-1, 1)
        self.delay_buffer = np.roll(self.delay_buffer, -frames)
        self.delay_buffer[-frames:] = indata.flatten()

    def run(self):
        with sd.Stream(channels=1, callback=self.callback, samplerate=self.samplerate):
            while 1: sd.sleep(100)

class PronounceApp(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Text to speech (with Echo)')
        self.setPlaceholderText("Type here and press Enter to listen...")
        self.echo_thread = EchoWorker()
        self.echo_thread.start()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            text = self.toPlainText().strip()
            if text:
                audio_file = "/tmp/tts.mp3"
                gTTS(text=text, lang='en', tld='us').save(audio_file)
                pygame.mixer.Sound(audio_file).play()
                #playsound(audio_file)
        else:
            super().keyPressEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PronounceApp()
    ex.show()
    sys.exit(app.exec_())
