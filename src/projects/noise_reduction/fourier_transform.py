import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from cupy.lib.stride_tricks import sliding_window_view

def compute_dft(chunk, N=512):
    # Discrete Fourier Transform following the 3Blue1Brown video from youtube (I wrote the code tho)
    # N is the index of sample in the ~32ms window
    # K is how many times the samples get rotated around the center of complex plane (the frequency we are checking)
    
    # K (num of bins) = N/2 + 1.
    # SR = Sampling Rate 
    # We want the real frequency f_r of measured k to be at most half the SR: f_r < SR/2
    # T1 is how long one sample lasts, that is 1/SR. T_n is how long N samples last, that is N * 1 / SR = N / SR.
    # 1 / T_n is how many samples of N fit in 1 second. f_r = k * (1 / T_n) = cycles per sample * samples per sec = k / T_n = k * SR / N
    # Going back: we want f_r < SR/2. This means k * SR / N < SR / 2 <=> k / N < 1/2 <=> k < 1/2 * N. 
    # This way we know why we bins to be half of N + 1 (+1 bc we account for 0)
    
    # Also why < N/2? Because we dont want aliasing, that is from fact that we need at least two points to tell the frequency of the wave,
    # and if we have say 3 cycles in 4 samples we cant accurately tell the frequency, and a lower frequency wave leaks into the other bins.
    # Intuition: If we snapshot once per N so thats k = 16000Hz and wave = 16000Hz then the wave is always in the same place.
    
    # Output for DFT is sum (g(t)*(e^-2i * pi * k * n / N)) = Fourier(k)
    # We sum over ns. They are samples. The n / N makes the furthest sample go up to 2pi * k on the complex circle.
    
    K = N // 2 + 1

    n = cp.arange(N) # (N,)
    k = cp.arange(K) # (K,)

    exponents = cp.outer(n, k) / N # outer product of shape (n, k)
    exponents = -exponents * 2j * cp.pi # exp[n, k] = the formula in line 8 without sum - sample n for k'th frequency

    W = cp.exp(exponents) # (N, K)
    
    dft = chunk @ W # (N,) @ (N, K) so we get just, K which are frequencies for current chunk
    
    return dft # (K,), an array of complex numbers as the approx of offset from the center of the circle.

    # Actually the sum is just half the energy, because of mirroring. We adjust for that when inverting the DFT.
    # We have to multiply the DFT by 2 everytime we need its magnitude. It is realistically two times bigger. 

def compute_dft_inv(dft_complex, N=512):
    # Logic of inverse: We have X[k] which is all information about a wave. It is a coordinate in the complex plane. It has a phase (angle between Re and Im)
    # and it has a magnitude (just sqrt(a^2 + b^2) from a + bi)
    # When we multiply two numbers in the complex plane, their magnitudes get scaled, and the phases add up.
    # The phase of X[k] is right, but the magnitude is N times too big. We added N points during calculations, driving the point somewhere.
    
    # We want ALL waves at EVERY frequency at time n to be summed up - we get the total wave out of sum of the ones it was broken into
    # To do this we have magnitude * e^(i * phase) * e^(i * 2pi * kn/N) - correspondingly X[k] * Timer.
    # Equivalent of operation: magnitude * e^i(2pi * kn/N + phase) -> rotate by whats inside of parenthesis.
    # The X[k] is info about the wave. The output of operation above is a:
    # vector moved by (phase + timer) counter-clockwise, scaled by magnitude. We know that wave should have done Timer cycles by time n,
    # so we just rotate it here and take its value.
    
    # We sum the values for each n for each k. This is the combined wave. We also divide by N because magnitude is N times too big.
    
    # Audio is 1D - but we get 2D numbers in complex plane. We just take the real part, because particles move only in the real realm. 
    
    # Note: Why real part of coordinates are the wave, if its a circle? Because they move like a sine with f = k
    
    # We have to multiply volume by 2, because DFT splits energy of real-world wave into two virtual waves spinning in opposite directions. 
    # Its just a fact. I cant really explain it right now. It is something with conjugates that a + bi and a - bi.
    
    K = len(dft_complex)
    n = cp.arange(N)
    k = cp.arange(K)
    exponents = 2j * cp.pi * cp.outer(n, k) / N # (N, K) - for each sample we have K frequencies
    W_inv = cp.exp(exponents) # (N, K), our timer witohut phase
    # dft is (K,), just K frequencies - waves info
    # Sum(X[k] * Timer at n for wave k)
    dft_work = dft_complex.copy()
    dft_work[1:-1] *= 2
    reconstruction = W_inv @ dft_work # (N, K) @ (K,) = (N,)
    # W_inv[N, :] is timer: for frequency k in sample n, where is the wave?
    # dft is pretty much the wave. In result we have for each frequency the product of the wave and timer (where to take the value from), we sum to get reconstruction
    # We also divide by N to adjust for huge magnitude of dft (we just added and didn't take the mean of the center of mass coordinates)

    audio_out = (cp.real(reconstruction)) / N
    
    return audio_out

def filter_downsample(audio, SR_orig, SR_target):
    # N = SR_orig * duration
    # R = SR_orig / SR_target <=> 48000Hz / 16000Hz = 3
    # Value at index i of 16kHz corresponds to i * R of original
    # Because i * R is often not an integer (R is not int) we interpolate, taking weighted averages of int(i * R) and int(i * R) + 1 (say its 2.75 we take 2 and 3)

    # In order to do slicing, we first do filtering. 
    # We do DFT(N=N, K=N/2), to get all possible frequencies from 0 to N /2, capturing whole spectrum of possible frequencies from the audio.

    # We want to cut off everything on the DFT higher than target_cutoff = SR_target / 2. That is 8kHz for 16kHz target, so the speech we reconstruct later has no aliasing.
    # We know that real frequency f_r = k * SR_orig / N, and k is integers so diff between them is SR_orig / N. That is our jump between bins.
    # k_cutoff = target_cutoff / jump = (SR_target / 2) / (SR_orig / N) = SR_target / SR_orig * N / 2 = N / 2R
    # As a result we get a DFT of frequencies from 0 to SR_target / 2. So for 16kHz we get from - to 16kHz, at SR = SR_original#
    # Audio is just (N,)
    N = audio.shape[0]
    R = SR_orig / SR_target
    
    dft_complex = compute_dft(audio, N)
    # dft is (K,), spectrum of frequencies in audio at SR_orig
    
    K_cutoff = int(N / (2*R)) # So the biggest frequency is SR_target / 2
    
    dft_filtered = dft_complex.copy()
    dft_filtered[K_cutoff:] = 0 + 0j 
    
    filtered_audio = compute_dft_inv(dft_filtered, N) # shape (N,)
    
    num_samples_target = int(N / R) # N * SR_target / SR_orig
    
    new_indices = cp.arange(num_samples_target) * R # Each sample gets mapped to sample in original, but we account for scale
    # new_indices go from 0 to N - 1 (our new N=target_samples)
    int_parts = new_indices.astype(int)
    fractions = new_indices - int_parts
    padded_audio = cp.append(filtered_audio, filtered_audio[-1]) # We double least sample, but dont add it to indices
    downsampled_audio = (1 - fractions) * padded_audio[int_parts] + fractions * padded_audio[int_parts + 1] # last sample is just the last sample if we get to N
    # Take the fraction part of first index, then fraction of second, based on where the index is.
    
    return downsampled_audio

def hann_window(N):
    n = cp.arange(N)
    
    return 0.5 * (1 - cp.cos(2 * cp.pi * n / (N-1))) # exactly 0 at n = 0, and 0 at n-1, 1 in the middle
    # When you add 2 Hann windows overlapped by 50% then they sum to a flat line.
    # Intuition is that if end of window 1 is 0, then if you move by 50% in the second window this end will be middle, and will be 1.
    # If you sum these up you get original.


def compute_stft(audio, N=512, hop=256):
    pad = N // 2 # Hann window zeroes out the edges that dont get compensated
    audio_padded = cp.pad(audio, (pad, pad), mode='reflect') # the reflect makes the audio a smoother wave, supposedly better than padding with 0's
    orig_len = len(audio)
    num_frames = (len(audio_padded) - N) // hop + 1 # math works out - amount of times we apply the window, also ensures we cover the last signal.
    # Even if the +1 overshoots, we land in padding. Say last sample ends on 1468 and whole is 1500 samples long - we calculate, and then cut out the padding in inverse.
    # The padding that was not used (the 32 samples from 1500 - 1468) is never used.
    K = N // 2 + 1
    spectrogram = cp.zeros((num_frames, K), dtype=complex)
    
    window = hann_window(N)
    
    for i in range(num_frames):
        start = i * hop 
        end = start + N
        chunk = audio_padded[start:end] # num_chunks = num_frames <=> chunk = frame
        
        spectrogram[i, :] = compute_dft(chunk * window, N)
        
    return spectrogram, orig_len # spectro is (num_frames, N)

def compute_stft_vectorized(audio, N=512, hop=256):
    audio = cp.asarray(audio)
    orig_len = len(audio)
    pad = N // 2
    audio_padded = cp.pad(audio, (pad, pad), mode='reflect')
    
    frames = sliding_window_view(audio_padded, N)[::hop] # creates a 2D matrix of shape (num_frames, N)
    # Pretty much just slices it all for us, using this simple line. Also is way faster. We could go over num_frames with for loop, but we do one big matmul.
    
    window = hann_window(N)
    windowed_frames = frames * window # Broadcasts window of shape (N,) over (num_frames, N)
    
    spectrogram = compute_dft(windowed_frames, N) # DFT for each chunk does the same thing -> gets frequencies. 
    # This means we can just put in a matrix (bc it uses matmul @) and it will apply the operation to each frame. 
    # The shapes are a bit diff in what we pass to DFT, because they have (num_frames, N) not (N,). It still works.
    
    return spectrogram, orig_len
    

def compute_stft_inv(spectrogram, orig_len, N=512, hop=256):
    num_frames = spectrogram.shape[0]
    
    total_samples = (num_frames - 1) * hop + N # with padding
    reconstructed_audio = cp.zeros(total_samples)
    window_sum = cp.zeros(total_samples)
    
    window = hann_window(N)
    
    for i in range(num_frames):
        chunk_rebuilt = compute_dft_inv(spectrogram[i, :], N)
        
        start = i * hop
        end = start + N
        
        reconstructed_audio[start:end] += chunk_rebuilt * window # We multiply again for protection. Explanation below
        window_sum[start:end] += window ** 2 # we multiplied by window once in stft, the once in inv stft. We make up for that by dividing later.
        # w(n) = 0.5 - 0.5(cos(2pi*n/N-1)) = sin^2(pi*n/(N-1))
        # We do w_first(n) + w_second(n)
        # If we move it by 50%, we get sin(x + 1/4th cycle) which is cos(x). So we get sin^2(x) + cos^2(x) = 1.0
        
        # If we do double: w(n)^2 we get w_first(n)^2 + w_second(n)^2 = sin^4(x) + cos^4(x) = 1 - 0.5sin^2(2x)
        # This means the value of window is a wave. So we dont have a constant, but a value for each sample
        
        # Why do we multiply by window the second time? 
        # We pass to the DFT already hanned sequence. This means fourier finds some combination of frequencies that zeros the audio out at the start and at the end.
        # When we modify it - remove noise frequency, etc. the end and start may no longer be 0 - the perfect balance of sum of frequencies is disturbed
        # This is why we apply the Hann window again - we smooth out the error noise at the ends. We got pretty much just the value from the middle, bc hann is tiny like 0.001
        # We divide later by SUM not PRODUCT of windows, so approximately by the middle ~ 1 but squared. The 0.001 almost vanishes.
        # So we do NOT go back to the original. We fitlered out the noise.
        
        # Why do we split Hann in the first place?
        # Without Hann, the audio we create back would have clicks. If we modify chunks, the wave being output no longer matches the input, 
        # and so if we glue them together they are not so smooth. This is because we modified the frequencies.
        # If we apply Hann window it is indeed smoother and has no jumps. It modifies the audio, because if we glued raw ISTFT it would have these bumps between samples.
        
    window_sum[window_sum < 1e-10] = 1.0 # so we dont div by 0
    reconstructed_audio /= window_sum
    
    pad_amount = N // 2
    return reconstructed_audio[pad_amount : pad_amount + orig_len]

def convert_to_db(spectrogram, N, is_raw_magnitude):
    if not is_raw_magnitude:
        mag = cp.abs(spectrogram)
    else:
        mag = spectrogram
        
    mag_norm = mag * 2 / N
    
    db_spectrogram = 20 * cp.log10(mag_norm + 1e-9) # we do 20 instead of 10 because we have amplitude and now power, and amplitude = power^2, so we bring 2 to the front
    # Our reference (bc the inside of log10 is mag/reference) is 1. This means if mag > 1 then spectro val will be > 0, but for now 1 is 0, 0.1 is -20, etc. It is unlikely that magnitude will go past 1 for some reason.
    
    return cp.maximum(db_spectrogram, -100) # Humans cant hear anything below -100dB
        
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
        vmin=-100, # black is whatever is less than -80dB.
        vmax=0 # At most 0dB
    )
    
    # Add the color scale legend
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Loudness (dB)')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)

    plt.show()