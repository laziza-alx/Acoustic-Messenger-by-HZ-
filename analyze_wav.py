
import wave
import numpy as np

# Our original message
original_bits = "01001101011010100110101011001010"

# Settings
filename = "fsk-audio.wav"
bit_duration = 0.1
f0 = 1000
f1 = 1200

# Read WAV file
with wave.open(filename, "rb") as wav:
    fs = wav.getframerate()
    channels = wav.getnchannels()
    sample_width = wav.getsampwidth()
    raw = wav.readframes(wav.getnframes())

# Convert audio to NumPy array
if sample_width == 2:
    signal = np.frombuffer(raw, dtype="<i2").astype(float)
elif sample_width == 1:
    signal = np.frombuffer(raw, dtype=np.uint8).astype(float) - 128
else:
    raise ValueError("Unsupported WAV sample width")

# Stereo to mono
if channels == 2:
    signal = signal.reshape(-1, 2).mean(axis=1)

samples_per_bit = round(fs * bit_duration)
print("Audio duration:", len(signal) / fs)
print("Expected duration:", len(original_bits) * bit_duration)
# Check actual frequencies in the first 8 bits
for i in range(32):
    start = i * samples_per_bit
    end = start + samples_per_bit
    segment = signal[start:end]

    spectrum = np.abs(np.fft.rfft(segment))
    frequencies = np.fft.rfftfreq(len(segment), d=1/fs)

    peak_index = np.argmax(spectrum[1:]) + 1
    detected_frequency = frequencies[peak_index]

    print(f"Bit {i + 1}: {detected_frequency:.1f} Hz")

# Decode one bit using FFT
def decode_bit(segment):
    N = len(segment)
    spectrum = np.abs(np.fft.rfft(segment))
    frequencies = np.fft.rfftfreq(N, d=1/fs)

    index_0 = np.argmin(np.abs(frequencies - f0))
    index_1 = np.argmin(np.abs(frequencies - f1))

    return "0" if spectrum[index_0] > spectrum[index_1] else "1"

# Decode all bits
decoded_bits = ""

for i in range(len(original_bits)):
    start = i * samples_per_bit
    end = start + samples_per_bit
    segment = signal[start:end]

    if len(segment) != samples_per_bit:
        raise ValueError("WAV is too short")

    decoded_bits += decode_bit(segment)

# Calculate BER
errors = sum(a != b for a, b in zip(original_bits, decoded_bits))
ber = errors / len(original_bits)

print("Sample rate:", fs)
print("Original bits:", original_bits)
print("Decoded bits: ", decoded_bits)
print("Bit errors:", errors)
print(f"BER: {ber:.2%}")
