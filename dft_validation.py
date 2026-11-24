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
    + 2 * np.sin(2 * np.pi * 1500 * t + np.pi / 4)
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
# Calculate phase in degrees
phase_dft = np.angle(X_dft[:N//2 + 1], deg=True)
phase_fft = np.angle(X_fft[:N//2 + 1], deg=True)

# Ignore phase of negligible frequency components
mask = magnitude_fft > 1e-8

# Compare phase spectra
fig, axes = plt.subplots(2, 1, figsize=(10, 7))

axes[0].stem(frequencies[mask], phase_dft[mask])
axes[0].set_title("Direct DFT Phase Spectrum")
axes[0].set_ylabel("Phase (degrees)")
axes[0].grid(True)

axes[1].stem(frequencies[mask], phase_fft[mask])
axes[1].set_title("NumPy FFT Phase Spectrum")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Phase (degrees)")
axes[1].grid(True)

plt.tight_layout()

# Print phase values
for i in np.where(mask)[0]:
    print(f"Frequency: {frequencies[i]} Hz")
    print(f"DFT phase: {phase_dft[i]:.2f} degrees")
    print(f"FFT phase: {phase_fft[i]:.2f} degrees")
plt.show()