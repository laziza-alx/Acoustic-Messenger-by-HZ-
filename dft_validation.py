import numpy as np
import matplotlib.pyplot as plt
import time

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

# Numerical validation
print("Numerical Validation")
print("Results match:", np.allclose(X_dft, X_fft))
print("Maximum error:", np.max(np.abs(X_dft - X_fft)))

# Positive frequency bins
frequencies = np.fft.rfftfreq(N, d=1/fs)

print("Frequency resolution:", fs / N, "Hz")

# ========================================
# 1. Magnitude Spectrum Comparison
# ========================================

magnitude_dft = np.abs(X_dft[:N//2 + 1])
magnitude_fft = np.abs(X_fft[:N//2 + 1])

fig, axes = plt.subplots(2, 1, figsize=(10, 7))

axes[0].stem(frequencies, magnitude_dft)
axes[0].set_title("Direct DFT Magnitude Spectrum")
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("Magnitude")
axes[0].grid(True)

axes[1].stem(frequencies, magnitude_fft)
axes[1].set_title("NumPy FFT Magnitude Spectrum")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Magnitude")
axes[1].grid(True)

plt.tight_layout()

# ========================================
# 2. Phase Spectrum Comparison
# ========================================

phase_dft = np.angle(X_dft[:N//2 + 1], deg=True)
phase_fft = np.angle(X_fft[:N//2 + 1], deg=True)

# Ignore phase of negligible frequency components
mask = magnitude_fft > 1e-8

fig, axes = plt.subplots(2, 1, figsize=(10, 7))

axes[0].stem(frequencies[mask], phase_dft[mask])
axes[0].set_title("Direct DFT Phase Spectrum")
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("Phase (degrees)")
axes[0].grid(True)

axes[1].stem(frequencies[mask], phase_fft[mask])
axes[1].set_title("NumPy FFT Phase Spectrum")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Phase (degrees)")
axes[1].grid(True)

plt.tight_layout()

print("\nPhase Spectrum Validation")

for i in np.where(mask)[0]:
    print(f"Frequency: {frequencies[i]} Hz")
    print(f"DFT phase: {phase_dft[i]:.2f} degrees")
    print(f"FFT phase: {phase_fft[i]:.2f} degrees")

# ========================================
# 3. Spectral Leakage Experiment
# ========================================

# Frequency aligned with a DFT bin
signal_aligned = np.sin(2 * np.pi * 1000 * t)

# Frequency between two DFT bins
signal_unaligned = np.sin(2 * np.pi * 1050 * t)

X_aligned = direct_dft(signal_aligned)
X_unaligned = direct_dft(signal_unaligned)

magnitude_aligned = np.abs(X_aligned[:N//2 + 1])
magnitude_unaligned = np.abs(X_unaligned[:N//2 + 1])

fig, axes = plt.subplots(2, 1, figsize=(10, 7))

axes[0].stem(frequencies, magnitude_aligned)
axes[0].set_title("1000 Hz - Frequency Bin Aligned")
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("Magnitude")
axes[0].grid(True)

axes[1].stem(frequencies, magnitude_unaligned)
axes[1].set_title("1050 Hz - Spectral Leakage")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Magnitude")
axes[1].grid(True)

for ax in axes:
    ax.set_ylim(0, 45)

plt.tight_layout()

print("\nSpectral Leakage Experiment")
print("Aligned frequency: 1000 Hz")
print("Unaligned frequency: 1050 Hz")
print("Frequency bin spacing:", fs / N, "Hz")

# ========================================
# 4. Execution Time Comparison
# ========================================

repetitions = 10

start = time.perf_counter()

for _ in range(repetitions):
    direct_dft(signal)

dft_time = (time.perf_counter() - start) / repetitions

start = time.perf_counter()

for _ in range(repetitions):
    np.fft.fft(signal)

fft_time = (time.perf_counter() - start) / repetitions

print("\nExecution Time Comparison")
print(f"Direct DFT: {dft_time:.8f} seconds")
print(f"NumPy FFT: {fft_time:.8f} seconds")
print(f"Speedup: {dft_time / fft_time:.2f}x")

# Display all graphs
plt.show()