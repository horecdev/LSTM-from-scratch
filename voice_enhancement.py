import numpy as np
import librosa as lb

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

    n = np.arange(N) # (N,)
    k = np.arange(K) # (K,)

    exponents = np.outer(n, k) / N # outer product of shape (n, k)
    exponents = -exponents * 2j * np.pi # exp[n, k] = the formula in line 8 without sum - sample n for k'th frequency

    W = np.exp(exponents) # (N, K)
    
    dft = chunk @ W # (N,) @ (N, K) so we get just, K which are frequencies for current chunk
    
    return dft # (K,), an array of complex numbers as the approx of offset from the center of the circle.

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
    n = np.arange(N)
    k = np.arange(K)
    exponents = 2j * np.pi * np.outer(n, k) / N # (N, K) - for each sample we have K frequencies
    W_inv = np.exp(exponents) # (N, K), our timer witohut phase
    # dft is (K,), just K frequencies - waves info
    # Sum(X[k] * Timer at n for wave k)
    # We multiply by 2 because of the mirroring effect. I cant explain it. The mirroring doesnt exist on 0th bin and Nyquist (257) bin
    dft_complex[1:-1] *= 2
    reconstruction = W_inv @ dft_complex # (N, K) @ (K,) = (N,)
    # W_inv[N, :] is timer: for frequency k in sample n, where is the wave?
    # dft is pretty much the wave. In result we have for each frequency the product of the wave and timer (where to take the value from), we sum to get reconstruction
    # We also divide by N to adjust for huge magnitude of dft (we just added and didn't take the mean of the center of mass coordinates)

    audio_out = (np.real(reconstruction)) / N
    
    return audio_out

def create_spectrogram(dft, N): # dft is (K,) of complex numpy numbers, N is chunk size the dft was calculated of.
    dft = np.abs(dft) / N  # We take the pythagorean distance from the middle and take the mean of how many contributions to the sum there were
    

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
    
    new_indices = np.arange(num_samples_target) * R # Each sample gets mapped to sample in original, but we account for scale
    # new_indices go from 0 to N - 1 (our new N=target_samples)
    int_parts = new_indices.astype(int)
    fractions = new_indices - int_parts
    padded_audio = np.append(filtered_audio, filtered_audio[-1]) # We double least sample, but dont add it to indices
    downsampled_audio = (1 - fractions) * padded_audio[int_parts] + fractions * padded_audio[int_parts + 1] # last sample is just the last sample if we get to N
    # Take the fraction part of first index, then fraction of second, based on where the index is.
    
    return downsampled_audio


    

    

