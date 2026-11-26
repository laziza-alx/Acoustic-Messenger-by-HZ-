"""
dft_receiver.py -- Standalone Direct-DFT FSK Receiver (WAV-only)

Reads a pre-recorded WAV file, detects FSK message boundaries independently of
the original bit sequence, decodes each bit interval with Direct DFT
(`signal_analysis.py`) into {0, 1, '?', 'No signal'}, compares against an
optional ground-truth bit sequence, prints BER / erasure statistics, and saves
per-bit CSV, summary CSV, decoded text, and a 4-panel diagnostic plot under
`results/dft_receiver/`.

Completely independent from `transmitter.py`, `receiver.py`, and `plot_test.py`.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import signal_analysis as sa


SETTINGS_FILE = "last_settings_dft_rx.json"
OUT_DIR = Path("results") / "dft_receiver"
PLOTS_DIR = OUT_DIR / "plots"
SUMMARY_CSV = OUT_DIR / "dft_receiver_summary.csv"

DEFAULT_CONFIG = {
    "bit_duration": 0.1,
    "f0": 1000.0,
    "f1": 2000.0,
    "guard": sa.DEFAULT_GUARD,
    "offset_ms": 0.0,
    "band_bins": sa.DEFAULT_BAND_BINS,
    "min_power": sa.DEFAULT_MIN_POWER,
    "min_snr": sa.DEFAULT_MIN_SNR,
    "min_band_ratio": sa.DEFAULT_MIN_BAND_RATIO,
    "min_contrast": sa.DEFAULT_MIN_CONTRAST,
}


def load_settings(path=SETTINGS_FILE):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    except (OSError, ValueError):
        return dict(DEFAULT_CONFIG)


def save_settings(settings, path=SETTINGS_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def ask(question, default, cast=str):
    while True:
        suffix = f" [{default}]" if default != "" else ""
        answer = input(f"{question}{suffix}: ").strip().strip('"')
        if answer == "":
            return default
        try:
            return cast(answer)
        except ValueError:
            print("  Invalid value, please try again.")


def parse_original_bits(raw_input_value):
    """Accept either a '0'/'1' bitstring, a path to a .txt file, or empty string."""
    text = (raw_input_value or "").strip().strip('"')
    if not text:
        return ""
    candidate = Path(text)
    if candidate.is_file():
        content = candidate.read_text(encoding="utf-8").strip()
        bits = "".join(ch for ch in content if ch in ("0", "1"))
        if not bits:
            raise ValueError(f"No '0'/'1' bits found in file: {candidate}")
        return bits
    if all(ch in ("0", "1") for ch in text):
        return text
    raise ValueError("Original bits must contain only '0'/'1' or be a valid .txt file path.")


def open_file(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        print(f"Open plot manually: {path}")


def write_bits_csv(path, bit_results, comparison=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bit_index",
        "t_bit_start_s",
        "t_from_s",
        "t_to_s",
        "outcome",
        "symbol",
        "reason",
        "power_0",
        "power_1",
        "mag_0",
        "mag_1",
        "noise_floor",
        "snr",
        "in_band_ratio",
        "contrast",
        "peak_freq_hz",
        "original_bit",
        "comparison_status",
    ]
    per_pos = comparison["per_position"] if comparison else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, b in enumerate(bit_results):
            orig_bit = per_pos[idx]["original"] if idx < len(per_pos) else ""
            cmp_status = per_pos[idx]["status"] if idx < len(per_pos) else ""
            writer.writerow({
                "bit_index": b["bit_index"],
                "t_bit_start_s": round(b["t_bit_start"], 6),
                "t_from_s": round(b["t_from"], 6),
                "t_to_s": round(b["t_to"], 6),
                "outcome": b["outcome"],
                "symbol": b["symbol"],
                "reason": b["reason"],
                "power_0": f"{b['power_0']:.8e}",
                "power_1": f"{b['power_1']:.8e}",
                "mag_0": f"{b['mag_0']:.6f}",
                "mag_1": f"{b['mag_1']:.6f}",
                "noise_floor": f"{b['noise_floor']:.8e}",
                "snr": f"{b['snr']:.4f}",
                "in_band_ratio": f"{b['in_band_ratio']:.4f}",
                "contrast": f"{b['contrast']:.4f}",
                "peak_freq_hz": f"{b['peak_freq_hz']:.2f}",
                "original_bit": orig_bit,
                "comparison_status": cmp_status,
            })
        # Record missing trailing positions when original was longer than decoded.
        for idx in range(len(bit_results), len(per_pos)):
            writer.writerow({
                "bit_index": idx,
                "t_bit_start_s": "",
                "t_from_s": "",
                "t_to_s": "",
                "outcome": "Missing",
                "symbol": "",
                "reason": "beyond_detected_end",
                "power_0": "",
                "power_1": "",
                "mag_0": "",
                "mag_1": "",
                "noise_floor": "",
                "snr": "",
                "in_band_ratio": "",
                "contrast": "",
                "peak_freq_hz": "",
                "original_bit": per_pos[idx]["original"],
                "comparison_status": per_pos[idx]["status"],
            })


def append_summary_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def plot_decode_diagnostics(x, fs, dec_res, f0, f1, bit_duration, comparison, path, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(x), dtype=np.float64) / float(fs)
    bit_results = dec_res["bit_results"]
    n = len(bit_results)

    fig, ax = plt.subplots(4, 1, figsize=(12, 14))

    # 1. Time-domain waveform + independent boundaries + bit labels
    ax[0].plot(t, x, linewidth=0.45, color="steelblue")
    top = max(float(np.max(np.abs(x))) if len(x) else 1.0, 1e-6)
    if dec_res["found"]:
        t_start = dec_res["t_start"]
        t_end = dec_res["t_end"]
        edges = t_start + np.arange(n + 1) * bit_duration
        for e in edges:
            ax[0].axvline(e, color="gray", linewidth=0.35)
        ax[0].axvline(t_start, color="green", linewidth=1.2, label=f"Detected start = {t_start:.3f} s")
        ax[0].axvline(t_end, color="purple", linewidth=1.2, linestyle="--", label=f"Detected end = {t_end:.3f} s")

        per_pos = comparison["per_position"] if comparison else []
        for i, b in enumerate(bit_results):
            status = per_pos[i]["status"] if i < len(per_pos) else "unknown"
            if status in ("substitution", "uncertain", "no_signal", "extra") or b["symbol"] in ("?", "-"):
                color = "red"
            else:
                color = "black"
            ax[0].text(edges[i] + bit_duration / 2.0, top * 1.03, b["symbol"],
                       ha="center", va="bottom", fontsize=7.5, color=color)
        ax[0].legend(loc="lower right", fontsize=8)
    else:
        ax[0].text(0.5, 0.5, "No FSK message detected (No signal)",
                   transform=ax[0].transAxes, ha="center", va="center", color="red", fontsize=12)

    ax[0].set_ylim(-top * 1.15, top * 1.25)
    ax[0].set_title("Time-Domain Signal and Independently Detected Bit Intervals")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude")
    ax[0].grid(True, alpha=0.25)

    # 2. Spectrogram
    nfft = min(1024, max(64, len(x) // 4))
    noverlap = int(0.85 * nfft)
    with np.errstate(divide="ignore"):
        power, _, _, image = ax[1].specgram(x, NFFT=nfft, Fs=fs, noverlap=noverlap, cmap="magma")
    if np.any(power > 0):
        top_db = 10.0 * np.log10(np.max(power))
        image.set_clim(top_db - 80.0, top_db)
    ax[1].set_ylim(0, min(fs / 2.0, 1.5 * max(f0, f1)))
    if dec_res["found"]:
        for e in dec_res["t_start"] + np.arange(n + 1) * bit_duration:
            ax[1].axvline(e, color="white", linewidth=0.35, alpha=0.7)
    ax[1].set_title("Spectrogram (Frequency Content Over Time)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Frequency (Hz)")

    # 3. Per-bit Direct DFT band magnitudes
    if n > 0:
        idx = np.arange(n)
        m0 = np.array([b["mag_0"] for b in bit_results])
        m1 = np.array([b["mag_1"] for b in bit_results])
        scale = max(float(max(np.max(m0), np.max(m1))), 1e-12)
        ax[2].bar(idx - 0.2, m0 / scale, width=0.4, label=f"Near {f0:.0f} Hz (0)")
        ax[2].bar(idx + 0.2, m1 / scale, width=0.4, label=f"Near {f1:.0f} Hz (1)")
        ax[2].set_xticks(idx)
        ax[2].set_xticklabels([b["symbol"] for b in bit_results], fontsize=8)
        per_pos = comparison["per_position"] if comparison else []
        for i, lbl in enumerate(ax[2].get_xticklabels()):
            status = per_pos[i]["status"] if i < len(per_pos) else ""
            if status in ("substitution", "uncertain", "no_signal", "extra") or bit_results[i]["symbol"] in ("?", "-"):
                lbl.set_color("red")
        ax[2].set_ylim(0, 1.35)
        ax[2].legend(loc="upper right", ncol=2, fontsize=8)
    ax[2].set_title("Per-Bit Direct DFT Magnitude (Normalized); '?' = Uncertain, '-' = No signal")
    ax[2].set_xlabel("Bit Index / Decoded Symbol")
    ax[2].set_ylabel("Relative Magnitude")
    ax[2].grid(axis="y", alpha=0.25)

    # 4. Direct DFT magnitude spectra of representative intervals
    plotted_any = False
    for target_sym, label_name in (("0", "0 bit"), ("1", "1 bit"), ("?", "Uncertain (?)"), ("-", "No signal (-)")):
        match = next((b for b in bit_results if b["symbol"] == target_sym and len(b["freqs"]) > 0), None)
        if match is not None:
            ax[3].plot(match["freqs"], match["magnitude"], label=f"Bit #{match['bit_index']} ({label_name})")
            plotted_any = True
    ax[3].axvline(f0, color="gray", linestyle="--", linewidth=0.9, label=f"f0 = {f0:.0f} Hz")
    ax[3].axvline(f1, color="black", linestyle=":", linewidth=0.9, label=f"f1 = {f1:.0f} Hz")
    ax[3].set_xlim(0, min(fs / 2.0, 1.5 * max(f0, f1)))
    ax[3].set_title("Direct DFT Magnitude Spectrum of Representative Bit Intervals")
    ax[3].set_xlabel("Frequency (Hz)")
    ax[3].set_ylabel("One-Sided Amplitude")
    if plotted_any:
        ax[3].legend(loc="upper right", fontsize=8)
    ax[3].grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_receiver(
    wav_path,
    f0,
    f1,
    bit_duration,
    guard=sa.DEFAULT_GUARD,
    offset_ms=0.0,
    original_bits="",
    note="",
    band_bins=sa.DEFAULT_BAND_BINS,
    min_power=sa.DEFAULT_MIN_POWER,
    min_snr=sa.DEFAULT_MIN_SNR,
    min_band_ratio=sa.DEFAULT_MIN_BAND_RATIO,
    min_contrast=sa.DEFAULT_MIN_CONTRAST,
    open_plot=False,
):
    wav_file = Path(wav_path)
    if not wav_file.is_file():
        raise FileNotFoundError(f"WAV file not found: {wav_file}")

    fs, x = sa.load_wav(wav_file)
    if bit_duration <= 0 or f0 <= 0 or f1 <= 0 or f0 == f1 or max(f0, f1) >= fs / 2.0:
        raise ValueError(f"Invalid parameters for fs={fs} Hz: need 0 < f0 != f1 < {fs / 2.0:.1f} Hz and bit_duration > 0.")

    t_run_start = time.perf_counter()
    dec_res = sa.decode_signal_message(
        x,
        fs,
        f0=f0,
        f1=f1,
        bit_duration=bit_duration,
        guard=guard,
        offset_ms=offset_ms,
        window="hann",
        method="dft_vectorized",
        band_bins=band_bins,
        min_power=min_power,
        min_snr=min_snr,
        min_band_ratio=min_band_ratio,
        min_contrast=min_contrast,
    )
    decode_time_s = time.perf_counter() - t_run_start

    decoded_seq = dec_res["decoded_sequence"]
    comparison = sa.compare_bit_sequences(decoded_seq, original_bits) if original_bits else None

    print()
    print("=" * 78)
    print(f"WAV File        : {wav_file}  ({len(x) / fs:.3f} s, fs = {fs} Hz)")
    if dec_res["found"]:
        print(f"Detected Bounds : {dec_res['t_start']:.4f} s -> {dec_res['t_end']:.4f} s "
              f"(coarse {dec_res['t_coarse_start']:.4f}..{dec_res['t_coarse_end']:.4f} s, offset {offset_ms:+g} ms)")
    else:
        print("Detected Bounds : None (No signal)")
    print(f"Thresholds      : min_power={min_power:.1e}, min_snr={min_snr:g}, "
          f"min_band_ratio={min_band_ratio:g}, min_contrast={min_contrast:g}, guard={guard:g}")
    print(f"Decode Time     : {decode_time_s:.3f} s (Direct DFT)")
    print("-" * 78)
    print(f"{'bit':>4} {'t_start':>8} {'P(f0)':>10} {'P(f1)':>10} {'SNR':>8} "
          f"{'inBand':>7} {'cntr':>6} {'peakHz':>8} {'out':>9} {'reason':>22}")
    for b in dec_res["bit_results"]:
        print(f"{b['bit_index']:>4} {b['t_bit_start']:>8.3f} {b['power_0']:>10.2e} {b['power_1']:>10.2e} "
              f"{b['snr']:>8.1f} {b['in_band_ratio']:>7.2f} {b['contrast']:>6.2f} "
              f"{b['peak_freq_hz']:>8.1f} {b['outcome']:>9} {b['reason']:>22}")

    print("-" * 78)
    print(f"Decoded Sequence ({dec_res['n_bits_detected']} bits): {decoded_seq or '<none>'}")
    print(f"  ('0'/'1' = decided, '?' = uncertain, '-' = No signal)")

    if comparison is not None:
        print(f"Original Sequence ({comparison['total_original_bits']} bits): {original_bits}")
        status_bar = "".join(
            "." if p["status"] == "correct"
            else ("X" if p["status"] == "substitution"
                  else ("?" if p["status"] == "uncertain"
                        else ("-" if p["status"] == "no_signal"
                              else ("M" if p["status"] == "missing" else "+"))))
            for p in comparison["per_position"]
        )
        print(f"Bitwise Comparison        : {status_bar}")
        print("  Legend: .=correct  X=substitution  ?=uncertain  -=no_signal  M=missing  +=extra")
        def_ber_str = f"{comparison['definite_ber']:.4f}" if not np.isnan(comparison["definite_ber"]) else "N/A"
        ovr_err_str = f"{comparison['overall_error_rate']:.4f}" if not np.isnan(comparison["overall_error_rate"]) else "N/A"
        print(f"  Correct bits       : {comparison['correct_bits']}")
        print(f"  Substitutions (S)  : {comparison['substitutions']}")
        print(f"  Uncertain (?) (U)  : {comparison['uncertain_bits']}")
        print(f"  No signal (-) (Ns) : {comparison['no_signal_bits']}")
        print(f"  Missing bits (M)   : {comparison['missing_bits']}")
        print(f"  Extra bits (X)     : {comparison['extra_bits']}")
        print(f"  Definite BER       : {def_ber_str}   [S / (C + S)]")
        print(f"  Overall Error Rate : {ovr_err_str}   [(S + U + Ns + M + X) / L_orig]")

    stem = wav_file.stem
    if offset_ms:
        stem += f"_offset{offset_ms:g}ms"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decoded_txt = OUT_DIR / f"{stem}_decoded.txt"
    decoded_txt.write_text(decoded_seq, encoding="utf-8")

    bits_csv = OUT_DIR / f"{stem}_bits.csv"
    write_bits_csv(bits_csv, dec_res["bit_results"], comparison=comparison)

    plot_path = PLOTS_DIR / f"{stem}_decode.png"
    ber_badge = (
        f"DefBER={comparison['definite_ber']:.3f}, ErrRate={comparison['overall_error_rate']:.3f}"
        if comparison and not np.isnan(comparison["definite_ber"])
        else f"Decoded={dec_res['n_bits_detected']} bits"
    )
    title = f"Direct DFT Receiver | T={bit_duration:g}s, f0={f0:g}Hz, f1={f1:g}Hz | {ber_badge}"
    if note:
        title += f" | {note}"
    plot_decode_diagnostics(x, fs, dec_res, f0, f1, bit_duration, comparison, plot_path, title)

    summary_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "wav_file": str(wav_file),
        "note": note,
        "fs_hz": fs,
        "duration_s": round(len(x) / float(fs), 4),
        "bit_duration_s": bit_duration,
        "f0_hz": f0,
        "f1_hz": f1,
        "guard": guard,
        "offset_ms": offset_ms,
        "min_power": min_power,
        "min_snr": min_snr,
        "min_band_ratio": min_band_ratio,
        "min_contrast": min_contrast,
        "detected_start_s": round(dec_res["t_start"], 4),
        "detected_end_s": round(dec_res["t_end"], 4),
        "n_bits_detected": dec_res["n_bits_detected"],
        "n_bits_original": comparison["total_original_bits"] if comparison else "",
        "decoded_sequence": decoded_seq,
        "original_sequence": original_bits,
        "correct_bits": comparison["correct_bits"] if comparison else "",
        "substitutions": comparison["substitutions"] if comparison else "",
        "uncertain_bits": comparison["uncertain_bits"] if comparison else sum(b["symbol"] == "?" for b in dec_res["bit_results"]),
        "no_signal_bits": comparison["no_signal_bits"] if comparison else sum(b["symbol"] == "-" for b in dec_res["bit_results"]),
        "missing_bits": comparison["missing_bits"] if comparison else "",
        "extra_bits": comparison["extra_bits"] if comparison else "",
        "definite_ber": round(comparison["definite_ber"], 6) if comparison and not np.isnan(comparison["definite_ber"]) else "",
        "overall_error_rate": round(comparison["overall_error_rate"], 6) if comparison and not np.isnan(comparison["overall_error_rate"]) else "",
        "decode_time_s": round(decode_time_s, 4),
    }
    append_summary_csv(SUMMARY_CSV, summary_row)

    print("=" * 78)
    print(f"Saved decoded txt : {decoded_txt}")
    print(f"Saved per-bit csv : {bits_csv}")
    print(f"Saved summary csv : {SUMMARY_CSV}")
    print(f"Saved plot        : {plot_path}")

    if open_plot:
        open_file(plot_path)

    return dec_res, comparison


def build_parser():
    parser = argparse.ArgumentParser(description="Direct-DFT FSK Receiver for pre-recorded WAV files")
    parser.add_argument("--wav", type=str, help="Path to input WAV file")
    parser.add_argument("--bit-duration", type=float, help="Bit duration in seconds")
    parser.add_argument("--f0", type=float, help="Frequency for bit 0 in Hz")
    parser.add_argument("--f1", type=float, help="Frequency for bit 1 in Hz")
    parser.add_argument("--guard", type=float, help="Guard fraction at each bit edge (e.g. 0.1)")
    parser.add_argument("--offset-ms", type=float, help="Timing offset adjustment in ms")
    parser.add_argument("--bits", type=str, default="", help="Original bitstring or .txt path (optional, for BER)")
    parser.add_argument("--note", type=str, default="", help="Optional trial note")
    parser.add_argument("--band-bins", type=int, default=sa.DEFAULT_BAND_BINS)
    parser.add_argument("--min-power", type=float, default=sa.DEFAULT_MIN_POWER)
    parser.add_argument("--min-snr", type=float, default=sa.DEFAULT_MIN_SNR)
    parser.add_argument("--min-band-ratio", type=float, default=sa.DEFAULT_MIN_BAND_RATIO)
    parser.add_argument("--min-contrast", type=float, default=sa.DEFAULT_MIN_CONTRAST)
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open the saved plot")
    return parser


def main():
    args = build_parser().parse_args()
    s = load_settings()

    if args.wav:
        wav_path = args.wav
        bit_duration = args.bit_duration if args.bit_duration is not None else s["bit_duration"]
        f0 = args.f0 if args.f0 is not None else s["f0"]
        f1 = args.f1 if args.f1 is not None else s["f1"]
        guard = args.guard if args.guard is not None else s["guard"]
        offset_ms = args.offset_ms if args.offset_ms is not None else s["offset_ms"]
        original_bits = parse_original_bits(args.bits)
        note = args.note
        open_plot = not args.no_open
    else:
        print("=== Acoustic Messenger: DIRECT-DFT RECEIVER (WAV-only) ===")
        print("(Press Enter to keep the value shown in [ ])")
        wav_path = ask("WAV file path", "")
        while not wav_path:
            print("  Please enter a WAV file path.")
            wav_path = ask("WAV file path", "")
        bit_duration = ask("Bit duration (s)", s["bit_duration"], float)
        f0 = ask("f0, frequency for bit 0 (Hz)", s["f0"], float)
        f1 = ask("f1, frequency for bit 1 (Hz)", s["f1"], float)
        guard = ask("Guard interval fraction (0..0.4)", s["guard"], float)
        offset_ms = ask("Timing offset (ms)", s["offset_ms"], float)
        while True:
            raw_bits = ask("Original bit sequence or .txt path (Enter to skip BER)", "")
            try:
                original_bits = parse_original_bits(raw_bits)
                break
            except ValueError as exc:
                print(f"  {exc}")
        note = ask("Trial note (optional)", "")
        open_plot = True

    save_settings({
        **s,
        "bit_duration": bit_duration,
        "f0": f0,
        "f1": f1,
        "guard": guard,
        "offset_ms": offset_ms,
        "band_bins": args.band_bins,
        "min_power": args.min_power,
        "min_snr": args.min_snr,
        "min_band_ratio": args.min_band_ratio,
        "min_contrast": args.min_contrast,
    })

    run_receiver(
        wav_path=wav_path,
        f0=f0,
        f1=f1,
        bit_duration=bit_duration,
        guard=guard,
        offset_ms=offset_ms,
        original_bits=original_bits,
        note=note,
        band_bins=args.band_bins,
        min_power=args.min_power,
        min_snr=args.min_snr,
        min_band_ratio=args.min_band_ratio,
        min_contrast=args.min_contrast,
        open_plot=open_plot,
    )


if __name__ == "__main__":
    main()
