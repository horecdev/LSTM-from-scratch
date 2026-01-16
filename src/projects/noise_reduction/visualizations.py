import matplotlib.pyplot as plt
import numpy as np
from src.projects.noise_reduction.fourier_transform import convert_to_db

    
def plot_spectrogram(spectrogram, SR, N, hop, title, is_raw_magnitude=False):

    db_spec = convert_to_db(spectrogram, N, is_raw_magnitude) # I guess if we know the magnitude and can turn it into dB. I need more math foundation for that tho.

    # We want to plot bins on the x axis, and time on the y axis
    db_spec = db_spec.get().T
    
    fig, ax = plt.subplots(figsize=(10, 5))
    # We want instead of indices of samples to see the seconds
    duration = spectrogram.shape[0] * hop / SR 
    extent = [0, duration, 0, SR/2] # We want to define the left, right, bottom, top
    
    img = ax.imshow(
        db_spec, 
        origin='lower', # Put 0Hz at the bottom, not top
        aspect='auto',
        extent=extent,
        cmap='magma',
        vmin=-80, # black is whatever is less than -80dB.
        vmax=0 # At most 0dB
    )
    
    # Add the color scale legend
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Loudness (dB)')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)

    plt.show()
    
def plot_loss_mask(noisy_db, clean_db):
    # How much audio we want to preserve
    speech_to_keep = np.clip((clean_db + 80) / 80, 0, 1)
    # Diff (noise to remove)
    noise_to_remove = np.clip((noisy_db - clean_db) / 80, 0, 1)
    
    # Importance map
    W_raw = np.maximum(speech_to_keep, noise_to_remove) + 0.05
    # Normalize
    W = W_raw / np.mean(W_raw)
    
    fig, axes = plt.subplots(1, 4, figsize=(30, 5))
    
    # Show noisy
    im0 = axes[0].imshow(noisy_db.T, origin='lower', aspect='auto', cmap='magma', vmin=-80, vmax=0)
    axes[0].set_title("Noisy Speech")
    plt.colorbar(im0, ax=axes[0])
    
    # Show clean
    im1 = axes[1].imshow(clean_db.T, origin='lower', aspect='auto', cmap='magma', vmin=-80, vmax=0)
    axes[1].set_title("Clean Speech")
    plt.colorbar(im0, ax=axes[1])
    
    # Noise to remove
    im1 = axes[2].imshow(noise_to_remove.T, origin='lower', aspect='auto', cmap='magma')
    axes[2].set_title("Noise To Remove")
    plt.colorbar(im1, ax=axes[2])
    
    # Final Loss Mask
    im2 = axes[3].imshow(W.T, origin='lower', aspect='auto', cmap='magma')
    axes[3].set_title("Loss Mask")
    plt.colorbar(im2, ax=axes[3])
    
    for ax in axes:
        ax.set_xlabel("Time (Frames)")
        ax.set_ylabel("Freq (Bins)")
        
    plt.tight_layout()
    plt.savefig('loss_mask.png')
    plt.show()

def plot_denoising_comparison(n_spec, c_spec, pred_mask, sr, N, hop):
    # Convert complex specs to dB
    n_db = convert_to_db(n_spec, N, is_raw_magnitude=False)
    c_db = convert_to_db(c_spec, N, is_raw_magnitude=False)
    mask = pred_mask
    
    duration = n_db.shape[0] * hop / sr
    extent = [0, duration, 0, sr/2]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Noisy
    im0 = axes[0].imshow(n_db.T, origin='lower', aspect='auto', extent=extent, cmap='magma', vmin=-80, vmax=0)
    axes[0].set_title("Original")
    plt.colorbar(im0, ax=axes[0])
    
    # Cleaned
    im1 = axes[1].imshow(c_db.T, origin='lower', aspect='auto', extent=extent, cmap='magma', vmin=-80, vmax=0)
    axes[1].set_title("Cleaned")
    plt.colorbar(im1, ax=axes[1])
    
    # Mask
    im2 = axes[2].imshow(mask.T, origin='lower', aspect='auto', extent=extent, cmap='binary_r', vmin=0, vmax=1)
    axes[2].set_title("Predicted Mask")
    plt.colorbar(im2, ax=axes[2])
    
    for ax in axes:
        ax.set_ylabel("Frequency (Hz)")
    axes[2].set_xlabel("Time (seconds)")
    
    plt.tight_layout()
    plt.savefig('denoising_comparison.png')
    plt.show()