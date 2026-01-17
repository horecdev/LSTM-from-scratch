import numpy as np
import librosa as lb
import soundfile as sf

from fourier_transform import filter_downsample

sr_target = 16000
path_to_wav = r"C:\Users\horec\Downloads\Noisy Sample.mp4"
target_path = r"C:\Users\horec\Downloads\test2.wav"

audio, _ = lb.load(path_to_wav, sr=sr_target, mono=True)

# If our downsampler didnt need 117Gb of RAM then we would use it
# audio, sr_orig = lb.load(path_to_wav, sr=None)
# audio = filter_downsample(audio, sr_orig, sr_target)

sf.write(target_path, audio, sr_target)