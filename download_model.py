from basic_pitch.inference import predict
import numpy as np

# Create a tiny audio sample to force model download
dummy_audio = np.zeros((16000,))  # 1 second of silence at 16kHz
predict(dummy_audio, 16000)
print("Basic-pitch model downloaded successfully")
