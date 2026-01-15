import os
import time
import cupy as cp
import numpy as np
import librosa as lb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Import the logic we already wrote
from fourier_transform import compute_stft_vectorized, convert_to_db

# Configuration
original_path = r"C:\Datasets\Clean and Noisy Audio Dataset"
target_path = r"C:\Datasets\Clean and Noisy Audio Dataset 16kHz NumPy Spectrogram Magnitudes"
sr_target = 16000
duration = 2
N = 512
hop = 256

def process_single_file(task):
    src, dst, sr_target, N, hop = task
    
    save_path = dst.replace('.wav', '.npy')
    if os.path.exists(save_path):
        return True
    try:
        # Loaded audio is on CPU, we gotta move later
        audio, _ = lb.load(src, sr=sr_target)
        
        target_samples = sr_target * duration
        if len(audio) > target_samples: # Make it exactly 2s with padding
            audio = audio[:target_samples]
        else:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        
        audio_gpu = cp.asarray(audio)
        
        spec_gpu, _ = compute_stft_vectorized(audio_gpu, N, hop)
        
        mag_gpu = cp.abs(spec_gpu) # We need only magnitude to calculate the mask and the x_input via decibel transition

        save_path = dst.replace('.wav', '.npy')
        np.save(save_path, mag_gpu.get().astype(np.float64)) # move to NumPy
        return True
    except Exception as e:
        return f"Error {src}: {str(e)}"

if __name__ == "__main__":
    # Collect files to convert into spectros
    all_tasks = []
    for split_orig, split_target in [('trainset_56spk', 'train'), ('testset', 'test')]:
        for cls in ['clean', 'noisy']:
            src_dir = os.path.join(original_path, f"{cls}_{split_orig}_wav")
            dest_dir = os.path.join(target_path, split_target, cls)
            os.makedirs(dest_dir, exist_ok=True)
            
            if not os.path.exists(src_dir): continue
                
            for filename in os.listdir(src_dir):
                if filename.endswith('.wav'):
                    all_tasks.append((os.path.join(src_dir, filename), 
                                      os.path.join(dest_dir, filename), 
                                      sr_target, N, hop))

    print(f"Total files: {len(all_tasks)}")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(process_single_file, all_tasks), # map func output to each task
                  total=len(all_tasks), 
                  desc="Preprocessing"))
        
    end_time = time.time()

    print(f"Time: {(end_time - start_time)/60:.2f} mins")