import os
import time
import cupy as cp
import numpy as np
import librosa as lb
from tqdm import tqdm

from fourier_transform import compute_stft_vectorized

original_path = r"D:\Datasets\Clean and Noisy Audio Dataset"
target_path = r"D:\Datasets\Magnitude_test" 
sr_target = 16000
duration = 2
N = 512
hop = 256

def process_single_file(task, pbar):
    src, dst, sr_target, N, hop = task
    # save as .npy not .wav
    save_path = dst.replace('.wav', '.npy')
    
    # Skip if already exists and has some size
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000: # 1KB min
        return True
    
    try:
        # Load audio files with our target SR
        audio, _ = lb.load(src, sr=sr_target)
        target_samples = sr_target * duration
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        else:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        
        # Run the spectro calc
        audio_gpu = cp.asarray(audio)
        spec_gpu, _ = compute_stft_vectorized(audio_gpu, N, hop)
        mag_cpu = cp.abs(spec_gpu).get().astype(np.float32) # also squash to float 32
        
        # Save directly to final path
        np.save(save_path, mag_cpu)
        
        del audio_gpu, spec_gpu, mag_cpu
        return True

    except Exception as e:
        # Cleanup partial file if crash occurs during save
        if os.path.exists(save_path):
            os.remove(save_path)
        pbar.write(f"ERROR: {os.path.basename(src)} -> {str(e)}")
        return False

def main():
    all_tasks = []
    splits = [('trainset_56spk', 'train'), ('testset', 'val')] # split map
    classes = ['clean', 'noisy']
    
    for split_orig, split_target in splits:
        for cls in classes:
            src_dir = os.path.join(original_path, f"{cls}_{split_orig}_wav")
            dest_dir = os.path.join(target_path, split_target, cls)
            
            if not os.path.exists(src_dir):
                continue
            
            os.makedirs(dest_dir, exist_ok=True)
            for filename in os.listdir(src_dir):
                if filename.endswith('.wav'):
                    all_tasks.append((
                        os.path.join(src_dir, filename),
                        os.path.join(dest_dir, filename),
                        sr_target, N, hop
                    ))

    print(f"Total files: {len(all_tasks)}")
    if not all_tasks:
        print("No files found. Check your paths.")
        return

    start_time = time.time()
    successes = 0
    
    pbar = tqdm(all_tasks, desc="Preprocessing")
    for i, task in enumerate(pbar):
        if process_single_file(task, pbar):
            successes += 1
        
        # Every 200 we defragment
        if i % 200 == 0:
            cp.get_default_memory_pool().free_all_blocks()
            
    end_time = time.time()

    print(f"\nProcessed {successes}/{len(all_tasks)} files.")
    print(f"Total Time: {(end_time - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()