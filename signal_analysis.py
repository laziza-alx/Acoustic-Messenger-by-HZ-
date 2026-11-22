
import numpy as np


# Direct Discrete Fourier Transform
def direct_dft(signal):
    N = len(signal)
    result = np.zeros(N, dtype=complex)

    for k in range(N):
        for n in range(N):
            angle = -2 * np.pi * k * n / N

            result[k] += signal[n] * (
                np.cos(angle) + 1j * np.sin(angle)
            )

    return result


def detect_frequency(signal, fs):
    N = len(signal)

    # Use our own Direct DFT
    X = direct_dft(signal)

    # Positive frequency bins
    frequencies = np.fft.rfftfreq(N, d=1/fs)
    magnitude = np.abs(X[:N//2 + 1])

    # Ignore DC component
    magnitude[0] = 0

    # Find strongest frequency
    peak_index = np.argmax(magnitude)

    return frequencies[peak_index]


def decode_bit(signal, fs, f0=1000, f1=2000):
    N = len(signal)

    X = direct_dft(signal)
    frequencies = np.fft.rfftfreq(N, d=1/fs)
    magnitude = np.abs(X[:N//2 + 1])

    # Find bins closest to our two frequencies
    index_0 = np.argmin(np.abs(frequencies - f0))
    index_1 = np.argmin(np.abs(frequencies - f1))

    power_0 = magnitude[index_0] ** 2
    power_1 = magnitude[index_1] ** 2

    if power_0 > power_1:
        return 0
    else:
        return 1


# Test our DFT against NumPy FFT
if __name__ == "__main__":
    signal = np.array([1, 2, 3, 4], dtype=float)

    my_dft = direct_dft(signal)
    numpy_fft = np.fft.fft(signal)

    print("Direct DFT:")
    print(my_dft)

    print("\nNumPy FFT:")
    print(numpy_fft)

    print("\nResults match:",
          np.allclose(my_dft, numpy_fft))

