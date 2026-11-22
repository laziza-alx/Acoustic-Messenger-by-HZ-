
import numpy as np
import matplotlib.pyplot as plt

from signal_analysis import direct_dft, detect_frequency, decode_bit

# Sampling settings
fs = 8000
N = 80

# Time axis
t = np.arange(N) / fs

# Generate two sine waves
signal_1 = np.sin(2 * np.pi * 1000 * t)
signal_2 = 2 * np.sin(2 * np.pi * 3000 * t)

# Combine them into one signal
signal = signal_1 + signal_2
print("Decoded bit:", decode_bit(signal, fs, 1000, 3000))
print("Detected:", detect_frequency(signal, fs), "Hz")

# Calculate Direct DFT
X = direct_dft(signal)

# Frequency axis
frequencies = np.fft.rfftfreq(N, d=1/fs)

# One-sided amplitude spectrum
magnitude = np.abs(X[:N//2 + 1]) * 2 / N
magnitude[0] /= 2
magnitude[-1] /= 2

# Create two plots in one window
fig, axes = plt.subplots(2, 1, figsize=(10, 7))

# Time Domain
axes[0].plot(t * 1000, signal, color="royalblue")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Time Domain - Combined Signal")
axes[0].grid(True)

# Frequency Domain
axes[1].stem(frequencies, magnitude)
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Amplitude")
axes[1].set_title("Frequency Domain - Direct DFT")
axes[1].grid(True)

plt.tight_layout()
plt.show()
