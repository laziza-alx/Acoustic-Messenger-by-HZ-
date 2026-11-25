"""
receiver.py  --  Acoustic Messenger (Person 2)

Records from the microphone (or reads a WAV file), finds where the message
starts, splits it into bit intervals, and decides each bit by comparing FFT
magnitudes near f0 and f1. Reports BER, saves a plot and logs every trial.

Run:  python receiver.py
It asks for the source (mic or a WAV file), bit duration, f0, f1, timing
offset and the bits that were sent. Press Enter to keep the value in [ ].
Recording from the microphone needs:  pip install sounddevice
"""

import csv
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_BITS = "10110010111000101101001110010110"
SETTINGS_FILE = "last_settings_rx.json"
MIC_FS = 44100
GUARD = 0.1


def load_wav(path):
    fs, x = wavfile.read(path)
    if x.dtype == np.uint8:
        x = (x.astype(float) - 128) / 128
    elif np.issubdtype(x.dtype, np.integer):
        x = x.astype(float) / np.iinfo(x.dtype).max
    else:
        x = x.astype(float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return fs, x - x.mean()


def interval_spectrum(x, fs, t_from, t_to):
    a = max(int(round(t_from * fs)), 0)
    b = min(int(round(t_to * fs)), len(x))
    segment = x[a:b]
    window = np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft(segment * window)) * 2 / window.sum()
    freqs = np.fft.rfftfreq(len(segment), 1 / fs)
    return freqs, spectrum


def peak_near(freqs, spectrum, target, width):
    band = np.abs(freqs - target) <= width
    return spectrum[band].max()


def find_message(x, fs, f0, f1, duration, frame=0.02, hop=0.001, smooth=0.03):
    times = np.arange(0, len(x) / fs - frame, hop)
    level = np.zeros(len(times))
    for i, t in enumerate(times):
        freqs, spectrum = interval_spectrum(x, fs, t, t + frame)
        width = freqs[1]
        level[i] = peak_near(freqs, spectrum, f0, width) + peak_near(freqs, spectrum, f1, width)

    taps = int(smooth / hop)
    level = np.convolve(level, np.ones(taps) / taps, mode="same")
    level_db = 20 * np.log10(level + 1e-12)
    peak_db = np.percentile(level_db, 95)
    digital_silence = level_db < peak_db - 100
    if digital_silence.mean() > 0.05:
        noise_db = peak_db - 60
    else:
        noise_db = np.percentile(level_db, 10)
    above = level_db > (noise_db + peak_db) / 2

    width = min(int(duration / hop), len(above))
    counts = np.convolve(above.astype(float), np.ones(width), mode="valid")
    i = int(np.argmax(counts))
    while i < len(above) - 1 and not above[i]:
        i += 1
    while i > 0 and above[i - 1]:
        i -= 1
    return times[i] + frame / 2


def measure_bits(x, fs, t_start, n_bits, bit_duration, f0, f1, guard):
    mags = np.zeros((n_bits, 2))
    for i in range(n_bits):
        t_from = t_start + (i + guard) * bit_duration
        t_to = t_start + (i + 1 - guard) * bit_duration
        if int((t_to - t_from) * fs) < 16 or t_from < 0 or t_from * fs >= len(x):
            continue
        freqs, spectrum = interval_spectrum(x, fs, t_from, t_to)
        width = freqs[1]
        mags[i, 0] = peak_near(freqs, spectrum, f0, width)
        mags[i, 1] = peak_near(freqs, spectrum, f1, width)
    return mags


def contrast(mags):
    return np.abs(mags[:, 1] - mags[:, 0]) / (mags.sum(axis=1) + 1e-12)


def refine_start(x, fs, t_coarse, n_bits, bit_duration, f0, f1, early=0.007, step=0.001):
    center = t_coarse + early
    candidates = np.arange(center - bit_duration / 2, center + bit_duration / 2, step)
    scores = [contrast(measure_bits(x, fs, t, n_bits, bit_duration, f0, f1, 0)).mean()
              for t in candidates]
    return candidates[int(np.argmax(scores))]


def decide(mags):
    strongest = mags.max(axis=1)
    typical = np.median(strongest)
    bits = ""
    for m0, m1 in mags:
        if max(m0, m1) < 0.25 * typical:
            bits += "?"
        elif m1 > m0:
            bits += "1"
        else:
            bits += "0"
    return bits


def bit_errors(decoded, truth):
    errors = sum(d != t for d, t in zip(decoded, truth)) + abs(len(decoded) - len(truth))
    return errors, errors / len(truth)


def plot_decode(x, fs, t_start, bits, mags, bit_duration, f0, f1, guard, truth, path, title):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.arange(len(x)) / fs
    n = len(bits)
    edges = t_start + np.arange(n + 1) * bit_duration
    wrong = [truth is not None and (i >= len(truth) or bits[i] != truth[i]) for i in range(n)]
    colors = ["red" if w else "black" for w in wrong]

    fig, ax = plt.subplots(4, 1, figsize=(12, 14))

    ax[0].plot(t, x, linewidth=0.4)
    for e in edges:
        ax[0].axvline(e, color="gray", linewidth=0.3)
    ax[0].axvline(t_start, color="red", linewidth=1, label=f"message start = {t_start:.3f} s")
    top = np.abs(x).max()
    for i, b in enumerate(bits):
        ax[0].text(edges[i] + bit_duration / 2, top * 1.02, b, ha="center",
                   va="bottom", fontsize=7, color=colors[i])
    ax[0].set_ylim(-top * 1.1, top * 1.2)
    ax[0].set_title("Received signal with the bit intervals used for decoding")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude")
    ax[0].legend(loc="lower right")

    with np.errstate(divide="ignore"):
        power, _, _, image = ax[1].specgram(x, NFFT=1024, Fs=fs, noverlap=896, cmap="magma")
    top_db = 10 * np.log10(power.max())
    image.set_clim(top_db - 80, top_db)
    ax[1].set_ylim(0, 1.5 * max(f0, f1))
    for e in edges:
        ax[1].axvline(e, color="white", linewidth=0.3)
    ax[1].set_title("Spectrogram (frequency content over time)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Frequency (Hz)")

    idx = np.arange(n)
    scale = mags.max() if mags.max() > 0 else 1
    ax[2].bar(idx - 0.2, mags[:, 0] / scale, width=0.4, label=f"near {f0:.0f} Hz (means 0)")
    ax[2].bar(idx + 0.2, mags[:, 1] / scale, width=0.4, label=f"near {f1:.0f} Hz (means 1)")
    ax[2].set_xticks(idx)
    ax[2].set_xticklabels(list(bits))
    for label, c in zip(ax[2].get_xticklabels(), colors):
        label.set_color(c)
    ax[2].set_title("FFT magnitude in each bit interval (normalized); red = wrong bit")
    ax[2].set_xlabel("Decoded bit, in order")
    ax[2].set_ylabel("Relative magnitude")
    ax[2].set_ylim(0, 1.35)
    ax[2].legend(loc="upper right", ncol=2)

    for value, name in (("0", "a 0 bit"), ("1", "a 1 bit")):
        if value in bits:
            i = bits.index(value)
            freqs, spectrum = interval_spectrum(x, fs, t_start + (i + guard) * bit_duration,
                                                t_start + (i + 1 - guard) * bit_duration)
            ax[3].plot(freqs, spectrum, label=f"bit #{i} ({name})")
    ax[3].axvline(f0, color="gray", linestyle="--", linewidth=0.8)
    ax[3].axvline(f1, color="gray", linestyle="--", linewidth=0.8)
    ax[3].set_xlim(0, 1.5 * max(f0, f1))
    ax[3].set_title(f"Magnitude spectrum of one interval of each kind (dashed: {f0:.0f} Hz and {f1:.0f} Hz)")
    ax[3].set_xlabel("Frequency (Hz)")
    ax[3].set_ylabel("|X(f)| (amplitude)")
    ax[3].legend(loc="upper right")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def log_trial(path, row):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def ask(question, default, cast=str):
    while True:
        answer = input(f"{question} [{default}]: ").strip().strip('"')
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


def record_microphone(seconds, fs):
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("Recording needs sounddevice: pip install sounddevice  "
                         "(or type the name of a WAV file instead of mic)")
    x = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return x[:, 0]


def open_file(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception:
        print(f"Open the plot yourself: {path}")


def main():
    s = load_settings(SETTINGS_FILE, {"bit_duration": 0.1, "f0": 1000.0, "f1": 2000.0})
    print("=== Acoustic Messenger: RECEIVER ===   (Enter = keep value in [ ])")
    source = ask("Source: mic, or a WAV file name", "mic")
    bit_duration = ask("Bit duration (s)", s["bit_duration"], float)
    f0 = ask("f0, frequency for bit 0 (Hz)", s["f0"], float)
    f1 = ask("f1, frequency for bit 1 (Hz)", s["f1"], float)
    offset_ms = ask("Timing offset (ms)", 0.0, float)
    while True:
        truth = ask("Bits that were sent (used only for BER)", DEFAULT_BITS)
        if all(b in "01" for b in truth):
            break
        print("  Use only 0 and 1.")
    note = ask("Trial note, e.g. 'distance 2 m, trial 1'", "")
    save_settings(SETTINGS_FILE, {"bit_duration": bit_duration, "f0": f0, "f1": f1})

    if source.lower() == "mic":
        seconds = len(truth) * bit_duration + 5
        input(f"Press Enter to start recording ({seconds:.1f} s), then PLAY on the transmitter...")
        print("Recording...")
        recording = record_microphone(seconds, MIC_FS)
        os.makedirs("audio", exist_ok=True)
        wav_path = os.path.join("audio", f"rec_{datetime.now():%Y%m%d_%H%M%S}.wav")
        wavfile.write(wav_path, MIC_FS, recording)
        print(f"Saved recording: {wav_path}")
    else:
        wav_path = source
        if not os.path.exists(wav_path):
            raise SystemExit(f"File not found: {wav_path}  (type mic to record instead)")

    fs, x = load_wav(wav_path)
    if bit_duration <= 0 or f0 == f1 or max(f0, f1) >= fs / 2:
        raise SystemExit(f"Invalid settings: need bit duration > 0, f0 != f1, both below {fs / 2:.0f} Hz.")

    n_bits = len(truth)
    t_coarse = find_message(x, fs, f0, f1, n_bits * bit_duration)
    t_start = refine_start(x, fs, t_coarse, n_bits, bit_duration, f0, f1) + offset_ms / 1000
    mags = measure_bits(x, fs, t_start, n_bits, bit_duration, f0, f1, GUARD)
    bits = decide(mags)
    errors, ber = bit_errors(bits, truth)

    print()
    print(f"File          : {wav_path}  ({len(x) / fs:.2f} s, fs = {fs} Hz)")
    print(f"Message start : {t_start:.4f} s (offset {offset_ms:+g} ms)")
    print()
    print(f"{'bit':>4} {'time (s)':>9} {'near ' + f'{f0:g}' + ' Hz':>14} "
          f"{'near ' + f'{f1:g}' + ' Hz':>14} {'decision':>9} {'sent':>5}")
    for i, (m0, m1) in enumerate(mags):
        mark = "" if bits[i] == truth[i] else "  <- error"
        print(f"{i:>4} {t_start + i * bit_duration:>9.3f} {m0:>14.4f} {m1:>14.4f} "
              f"{bits[i]:>9} {truth[i]:>5}{mark}")
    print()
    print(f"Decoded : {bits}")
    print(f"Sent    : {truth}")
    print("=" * 64)
    print(f"  BER = {ber:.4f}   ({errors} errors out of {n_bits} bits)")
    print(f"  of which {bits.count('?')} bits had neither tone (shown as ?)")
    print("=" * 64)

    stem = os.path.splitext(os.path.basename(wav_path))[0]
    if offset_ms:
        stem += f"_offset{offset_ms:g}ms"
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"{stem}_decoded.txt"), "w") as f:
        f.write(bits)
    title = (f"T = {bit_duration:g} s, f0 = {f0:g} Hz, f1 = {f1:g} Hz, offset = {offset_ms:g} ms"
             f"  |  BER = {errors}/{n_bits}" + (f"  |  {note}" if note else ""))
    plot_path = os.path.join("results", "plots", f"{stem}_decode.png")
    plot_decode(x, fs, t_start, bits, mags, bit_duration, f0, f1, GUARD, truth, plot_path, title)

    log_trial(os.path.join("results", "trials.csv"), {
        "time": datetime.now().isoformat(timespec="seconds"),
        "file": wav_path, "note": note, "fs": fs,
        "bit_duration_s": bit_duration, "f0_hz": f0, "f1_hz": f1, "offset_ms": offset_ms,
        "start_s": round(t_start, 4), "n_bits": n_bits, "decoded": bits, "sent": truth,
        "errors": errors, "ber": round(ber, 4),
    })
    print(f"Saved plot   : {plot_path}")
    print(f"Logged trial : {os.path.join('results', 'trials.csv')}")
    open_file(plot_path)


if __name__ == "__main__":
    main()
