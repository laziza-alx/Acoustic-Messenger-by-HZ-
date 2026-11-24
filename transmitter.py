"""
transmitter.py  --  Acoustic Messenger (Person 1)

Converts a bit string into an audio signal using binary FSK
(0 -> f0 Hz, 1 -> f1 Hz), saves it as a mono WAV file, plots the waveform,
and plays it through the speaker.

Run:  python transmitter.py
It asks for the bits, bit duration, f0 and f1. Press Enter to keep the value
shown in [ ] (the value used last time).
"""

import json
import os
import subprocess
import sys
import wave

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots to files; no window needed
import matplotlib.pyplot as plt

# Default 32-bit test message (change freely)
DEFAULT_BITS = "10110010111000101101001110010110"


def bits_to_signal(bits, bit_duration=0.1, f0=1000.0, f1=2000.0,
                   fs=44100, amplitude=0.8, lead_silence=0.5,
                   tail_silence=0.5, continuous_phase=True):
    """Return (signal, bit_start_times). `bits` is a string like '1011'."""
    if any(b not in "01" for b in bits):
        raise ValueError("bits must contain only '0' and '1'")

    samples_per_bit = int(round(bit_duration * fs))
    freqs = np.array([f1 if b == "1" else f0 for b in bits], dtype=float)
    freq_per_sample = np.repeat(freqs, samples_per_bit)

    if continuous_phase:
        # accumulate phase so there are no clicks at bit boundaries
        phase = 2 * np.pi * np.cumsum(freq_per_sample) / fs
    else:
        # restart each bit at phase 0
        n = np.arange(samples_per_bit) / fs
        phase = np.concatenate([2 * np.pi * f * n for f in freqs])

    tone = amplitude * np.sin(phase)

    lead = np.zeros(int(round(lead_silence * fs)))
    tail = np.zeros(int(round(tail_silence * fs)))
    signal = np.concatenate([lead, tone, tail])

    bit_starts = lead_silence + np.arange(len(bits)) * bit_duration
    return signal, bit_starts


def save_wav(path, signal, fs):
    """Save a float signal in [-1, 1] as 16-bit mono PCM WAV."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    pcm = np.int16(np.clip(signal, -1, 1) * 32767)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(pcm.tobytes())


def plot_waveform(signal, fs, bits, bit_starts, bit_duration, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    t = np.arange(len(signal)) / fs

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].plot(t, signal, linewidth=0.5)
    axes[0].set_title("Transmitted signal (full message)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    for i, b in enumerate(bits):
        axes[0].text(bit_starts[i] + bit_duration / 2, 0.95, b,
                     ha="center", va="top", fontsize=7)
        axes[0].axvline(bit_starts[i], color="gray", linewidth=0.3)
    axes[0].set_ylim(-1.05, 1.05)

    # zoom ~10 ms around the first bit change so the two tones are visible
    change = next((i for i in range(1, len(bits)) if bits[i] != bits[i-1]), 1)
    tc = bit_starts[change]
    mask = (t >= tc - 0.005) & (t < tc + 0.005)
    axes[1].plot(t[mask], signal[mask])
    axes[1].axvline(tc, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_title(f"Zoom: bit {change-1} ({bits[change-1]}) -> bit {change} ({bits[change]})")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


SETTINGS_FILE = "last_settings_tx.json"
FS = 44100


def ask(question, default, cast=str):
    while True:
        answer = input(f"{question} [{default}]: ").strip()
        if answer == "":
            return default
        try:
            return cast(answer)
        except ValueError:
            print("  Not a valid value, try again.")


def load_settings(path, defaults):
    try:
        with open(path) as f:
            return {**defaults, **json.load(f)}
    except (OSError, ValueError):
        return defaults


def save_settings(path, settings):
    with open(path, "w") as f:
        json.dump(settings, f)


def play_wav(path):
    if sys.platform.startswith("win"):
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME)
    elif sys.platform == "darwin":
        subprocess.run(["afplay", path])
    else:
        subprocess.run(["aplay", path])


def main():
    s = load_settings(SETTINGS_FILE, {"bit_duration": 0.1, "f0": 1000.0, "f1": 2000.0})
    print("=== Acoustic Messenger: TRANSMITTER ===   (Enter = keep value in [ ])")
    while True:
        bits = ask("Bits to send", DEFAULT_BITS)
        if all(b in "01" for b in bits):
            break
        print("  Use only 0 and 1.")
    bit_duration = ask("Bit duration (s)", s["bit_duration"], float)
    f0 = ask("f0, frequency for bit 0 (Hz)", s["f0"], float)
    f1 = ask("f1, frequency for bit 1 (Hz)", s["f1"], float)

    if bit_duration <= 0 or f0 == f1 or max(f0, f1) >= FS / 2 or min(f0, f1) <= 0:
        raise SystemExit(f"Invalid settings: need bit duration > 0, f0 != f1, "
                         f"both between 0 and {FS / 2:.0f} Hz.")
    save_settings(SETTINGS_FILE, {"bit_duration": bit_duration, "f0": f0, "f1": f1})

    name = f"tx_{bit_duration:g}s_{f0:g}Hz_{f1:g}Hz"
    wav_path = os.path.join("audio", name + ".wav")
    plot_path = os.path.join("results", "plots", name + "_waveform.png")

    signal, starts = bits_to_signal(bits, bit_duration, f0, f1, FS)
    save_wav(wav_path, signal, FS)
    plot_waveform(signal, FS, bits, starts, bit_duration, plot_path)
    with open(os.path.join("audio", name + "_bits.txt"), "w") as f:
        f.write(bits)

    print()
    print("=" * 64)
    print(f"  bit duration {bit_duration:g} s | f0 {f0:g} Hz | f1 {f1:g} Hz | {len(bits)} bits")
    print(f"  sound length {len(signal) / FS:.1f} s -> type the SAME values on the receiver")
    print("=" * 64)
    print(f"Saved {wav_path} and {plot_path}")

    while True:
        if input("Press Enter to PLAY (q to quit): ").strip().lower() == "q":
            break
        print("Playing...")
        play_wav(wav_path)
        print("Done.")


if __name__ == "__main__":
    main()
