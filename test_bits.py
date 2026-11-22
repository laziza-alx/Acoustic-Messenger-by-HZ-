
import numpy as np

from signal_analysis import decode_bit

fs = 8000
bit_duration = 0.01

f0 = 1000
f1 = 3000

bits = "0101"

samples_per_bit = int(fs * bit_duration)

# Generate BFSK signal
signal_parts = []

for bit in bits:
    t = np.arange(samples_per_bit) / fs

    frequency = f0 if bit == "0" else f1

    wave = np.sin(2 * np.pi * frequency * t)
    signal_parts.append(wave)

signal = np.concatenate(signal_parts)

# Decode each bit separately
decoded_bits = ""

print("Starting decoding...", flush=True)

for i in range(len(bits)):
    print(f"Decoding bit {i + 1}...", flush=True)

    start = i * samples_per_bit
    end = start + samples_per_bit

    segment = signal[start:end]

    detected_bit = decode_bit(segment, fs, f0, f1)
    decoded_bits += str(detected_bit)

    print(f"Finished bit {i + 1}: {detected_bit}", flush=True)

print("Original bits:", bits)
print("Decoded bits: ", decoded_bits)
print("Match:", bits == decoded_bits)