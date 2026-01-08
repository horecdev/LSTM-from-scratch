import numpy as np

def compute_spectrogram(chunk, N=512, K=257):
    # Discrete Fourier Transform following the 3Blue1Brown video from youtube
    # N is the index of sample in the ~32ms window
    # K is how many times the samples get rotated around the center of complex plane (the frequency we are checking)
    
    # Output for DFT is sum(e^-2i * pi * k * n / N)
    # We sum over ns. They are samples. The n / N makes the furthest sample go up to 2pi * k on the complex circle.

    n = np.arange(N) # (N,)
    k = np.arange(K) # (K,)

    exponents = np.outer(n, k) / N # outer product of shape (n, k)
    exponents = -exponents * 2j * np.pi # exp[n, k] = the formula in line 8 without sum

    W = np.exp(exponents)

