import numpy as np
import matplotlib.pyplot as plt

from signal_analysis import direct_dft

# Sampling settings
fs = 8000
N = 80

t = np.arange(N) / fs

# Generate a combined signal
signal = (
    np.sin(2 * np.pi * 1000 * t)
    + 2 * np.sin(2 * np.pi * 1500 * t)
)

# Calculate Direct DFT and NumPy FFT
X_dft = direct_dft(signal)
X_fft = np.fft.fft(signal)

# Compare numerical results
print("Results match:", np.allclose(X_dft, X_fft))
print("Maximum error:", np.max(np.abs(X_dft - X_fft)))

# Positive frequency bins
frequencies = np.fft.rfftfreq(N, d=1/fs)

# Calculate magnitudes
magnitude_dft = np.abs(X_dft[:N//2 + 1])
magnitude_fft = np.abs(X_fft[:N//2 + 1])

# Compare spectra
fig, axes = plt.subplots(2, 1, figsize=(10, 7))

axes[0].stem(frequencies, magnitude_dft)
axes[0].set_title("Direct DFT Spectrum")
axes[0].set_ylabel("Magnitude")
axes[0].grid(True)

axes[1].stem(frequencies, magnitude_fft)
axes[1].set_title("NumPy FFT Spectrum")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Magnitude")
axes[1].grid(True)

plt.tight_layout()
plt.show()