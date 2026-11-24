import sounddevice as sd
import numpy as np

samplerate = int(sd.query_devices(None, 'input')['default_samplerate'])
delay_buffer = np.zeros(samplerate, dtype=np.float32)

def callback(indata, outdata, frames, time, status):
    global delay_buffer
    outdata[:] = delay_buffer[:frames].reshape(-1, 1)
    delay_buffer = np.roll(delay_buffer, -frames)
    delay_buffer[-frames:] = indata.flatten()

stream = sd.Stream(channels=1, callback=callback, samplerate=samplerate)
with stream:
    input()
