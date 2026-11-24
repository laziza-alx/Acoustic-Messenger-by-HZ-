import numpy as np
from signal_analysis import decode_bit

fs = 8000
bit_duration = 0.01
f0 = 1000

samples_per_bit = int(fs * bit_duration)

test_cases = [
    ("0101", 3000),
    ("0000", 1200),
    ("1111", 1200),
    ("1010", 1200),
    ("01101001", 1200),
    ("01101001", 1500),
    ("01101001", 2000)
]

for bits, f1 in test_cases:
    decoded_bits = ""

    for bit in bits:
        t = np.arange(samples_per_bit) / fs

        frequency = f0 if bit == "0" else f1
        signal = np.sin(2 * np.pi * frequency * t)

        detected_bit = decode_bit(signal, fs, f0, f1)
        decoded_bits += str(detected_bit)

    errors = sum(
        a != b for a, b in zip(bits, decoded_bits)
    )

    print(f"Tone separation: {f1 - f0} Hz")
    print("Original bits:", bits)
    print("Decoded bits: ", decoded_bits)
    print("Match:", bits == decoded_bits)
    print(f"BER: {errors / len(bits):.2%}")
    print()