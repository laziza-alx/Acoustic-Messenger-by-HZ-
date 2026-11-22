
import numpy as np
import matplotlib.pyplot as plt

# Settings
fs = 8000
bit_duration = 0.1
f0 = 1000

separations = [200, 500, 1000]
bits = "01001101011010100110101011001010"

samples_per_bit = int(fs * bit_duration)


# Fast frequency-based bit decoder
def decode_fast(segment, f0, f1):
    N = len(segment)

    X = np.fft.rfft(segment)
    frequencies = np.fft.rfftfreq(N, d=1/fs)
    magnitude = np.abs(X)

    index_0 = np.argmin(np.abs(frequencies - f0))
    index_1 = np.argmin(np.abs(frequencies - f1))

    return 0 if magnitude[index_0] > magnitude[index_1] else 1


# Run one experiment trial
def run_trial(f1, noise_level, rng):
    decoded = ""

    for bit in bits:
        t = np.arange(samples_per_bit) / fs

        frequency = f0 if bit == "0" else f1

        # Generate sine wave
        wave = np.sin(2 * np.pi * frequency * t)

        # Add simulated noise
        noise = rng.normal(0, noise_level, samples_per_bit)
        received = wave + noise

        # Decode received bit
        detected = decode_fast(received, f0, f1)
        decoded += str(detected)

    # Calculate Bit Error Rate
    errors = sum(a != b for a, b in zip(bits, decoded))
    ber = errors / len(bits)

    return ber


# Run experiments
rng = np.random.default_rng(42)
results = []

for separation in separations:
    f1 = f0 + separation
    trial_bers = []

    for trial in range(30):
        ber = run_trial(f1, noise_level=5.0, rng=rng)
        trial_bers.append(ber)

    average_ber = np.mean(trial_bers)
    results.append(average_ber)

    print(f"Separation: {separation} Hz")
    print(f"Trial BERs: {trial_bers}")
    print(f"Average BER: {average_ber:.2%}")
    print()


# Plot results
plt.figure(figsize=(8, 5))
plt.plot(separations, results, marker="o")

plt.xlabel("Tone Separation (Hz)")
plt.ylabel("Average BER")
plt.title("Tone Separation vs Bit Error Rate")
plt.ylim(0, 1)
plt.grid(True)

plt.show()
