"""
dft_validation.py -- Independent Direct DFT vs. NumPy FFT Evaluation Suite

Evaluates the Direct DFT implementations in `signal_analysis.py` against
`np.fft.fft` across three controlled, reproducible experiment groups:
  A. Mathematical correctness (numerical error, amplitude & frequency accuracy)
  B. Spectral characteristics (resolution, leakage, rectangular vs Hann, off-bin)
  C. Practical performance (O(N^2) vs O(N log N) timing, SNR vs BER & detection rate)

Completely independent from `transmitter.py`, `receiver.py`, `dft_receiver.py`,
and `plot_test.py`. Saves CSV metrics and labeled PNG charts to
`results/dft_evaluation/`.
"""

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import signal_analysis as sa


OUT_DIR = Path("results") / "dft_evaluation"
PLOTS_DIR = OUT_DIR / "plots"
SEED = 42


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# Group A: Mathematical Correctness
# ============================================================================

def evaluate_mathematical_correctness():
    rng = np.random.default_rng(SEED)
    fs = 8000.0
    t80 = np.arange(80) / fs
    t85 = np.arange(85) / fs

    cases = [
        ("impulse_8", fs, np.array([1.0] + [0.0] * 7), None, None),
        ("constant_16", fs, np.ones(16) * 0.5, 0.0, 0.5),
        ("single_tone_aligned_1000Hz", fs, 0.8 * np.sin(2 * np.pi * 1000.0 * t80), 1000.0, 0.8),
        ("two_tone_phase_80", fs,
         0.6 * np.sin(2 * np.pi * 1000.0 * t80) + 1.2 * np.sin(2 * np.pi * 1500.0 * t80 + np.pi / 4),
         1500.0, 1.2),
        ("odd_length_tone_85", fs, 0.9 * np.sin(2 * np.pi * 1600.0 * t85), 1600.0, 0.9),
        ("noisy_tone_1200Hz", fs,
         0.75 * np.sin(2 * np.pi * 1200.0 * t80) + rng.normal(0, 0.05, 80),
         1200.0, 0.75),
    ]

    rows = []
    demo_freqs = demo_dft_mag = demo_fft_mag = demo_dft_phase = demo_fft_phase = None

    for name, case_fs, sig, true_f, true_amp in cases:
        X_loop = sa.direct_dft(sig)
        X_vec = sa.direct_dft_vectorized(sig)
        X_fft = np.fft.fft(sig)

        diff_loop = np.abs(X_loop - X_fft)
        diff_vec = np.abs(X_vec - X_fft)
        fft_norm = float(np.linalg.norm(X_fft))

        max_err_loop = float(np.max(diff_loop))
        mean_err_loop = float(np.mean(diff_loop))
        rel_err_loop = float(np.linalg.norm(X_loop - X_fft) / max(fft_norm, 1e-18))

        max_err_vec = float(np.max(diff_vec))
        mean_err_vec = float(np.mean(diff_vec))
        rel_err_vec = float(np.linalg.norm(X_vec - X_fft) / max(fft_norm, 1e-18))

        # Rectangular-window amplitude & peak frequency accuracy comparison
        freqs, mag_dft, _ = sa.compute_spectrum(sig, case_fs, window="rect", method="dft_vectorized")
        _, mag_fft, _ = sa.compute_spectrum(sig, case_fs, window="rect", method="fft")

        if true_f is not None and true_f > 0:
            est_f_dft = sa.detect_frequency(sig, case_fs, window="rect", method="dft_vectorized")
            est_f_fft = sa.detect_frequency(sig, case_fs, window="rect", method="fft")
            freq_err_dft = abs(est_f_dft - true_f)
            freq_err_fft = abs(est_f_fft - true_f)
            idx_pk = int(np.argmin(np.abs(freqs - true_f)))
            amp_dft = float(mag_dft[idx_pk])
            amp_fft = float(mag_fft[idx_pk])
            amp_err_dft = abs(amp_dft - true_amp)
            amp_err_fft = abs(amp_fft - true_amp)
        else:
            est_f_dft = est_f_fft = freq_err_dft = freq_err_fft = ""
            amp_dft = amp_fft = amp_err_dft = amp_err_fft = ""

        if name == "two_tone_phase_80":
            demo_freqs = freqs
            demo_dft_mag = mag_dft
            demo_fft_mag = mag_fft
            half = len(freqs)
            demo_dft_phase = np.angle(X_vec[:half], deg=True)
            demo_fft_phase = np.angle(X_fft[:half], deg=True)

        rows.append({
            "case": name,
            "N": len(sig),
            "fs_hz": case_fs,
            "max_abs_err_loop": f"{max_err_loop:.3e}",
            "mean_abs_err_loop": f"{mean_err_loop:.3e}",
            "rel_l2_err_loop": f"{rel_err_loop:.3e}",
            "max_abs_err_vec": f"{max_err_vec:.3e}",
            "mean_abs_err_vec": f"{mean_err_vec:.3e}",
            "rel_l2_err_vec": f"{rel_err_vec:.3e}",
            "true_freq_hz": true_f if true_f is not None else "",
            "dft_freq_hz": est_f_dft,
            "fft_freq_hz": est_f_fft,
            "freq_err_dft_hz": f"{freq_err_dft:.4f}" if freq_err_dft != "" else "",
            "true_amp": true_amp if true_amp is not None else "",
            "dft_amp": f"{amp_dft:.6f}" if amp_dft != "" else "",
            "fft_amp": f"{amp_fft:.6f}" if amp_fft != "" else "",
            "amp_err_dft": f"{amp_err_dft:.6f}" if amp_err_dft != "" else "",
        })

    save_csv(OUT_DIR / "math_correctness.csv", rows)

    # Plot Group A
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    names = [r["case"] for r in rows]
    x_pos = np.arange(len(names))
    loop_errs = [max(float(r["max_abs_err_loop"]), 1e-17) for r in rows]
    vec_errs = [max(float(r["max_abs_err_vec"]), 1e-17) for r in rows]

    axes[0, 0].bar(x_pos - 0.18, loop_errs, width=0.36, label="direct_dft (loop)")
    axes[0, 0].bar(x_pos + 0.18, vec_errs, width=0.36, label="direct_dft_vectorized")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[0, 0].set_ylabel("Max Absolute Error vs FFT")
    axes[0, 0].set_title("A1. Numerical Error vs Reference np.fft.fft")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    mean_errs = [max(float(r["mean_abs_err_loop"]), 1e-17) for r in rows]
    rel_errs = [max(float(r["rel_l2_err_loop"]), 1e-17) for r in rows]
    axes[0, 1].bar(x_pos - 0.18, mean_errs, width=0.36, label="Mean Abs Error")
    axes[0, 1].bar(x_pos + 0.18, rel_errs, width=0.36, label="Relative L2 Error")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[0, 1].set_ylabel("Error Magnitude")
    axes[0, 1].set_title("A2. Mean Absolute & Relative L2 Errors")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    axes[1, 0].stem(demo_freqs, demo_fft_mag, linefmt="C1-", markerfmt="C1o", basefmt=" ", label="NumPy FFT")
    axes[1, 0].stem(demo_freqs, demo_dft_mag, linefmt="C0--", markerfmt="C0x", basefmt=" ", label="Direct DFT")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("One-Sided Amplitude")
    axes[1, 0].set_title("A3. Magnitude Spectrum Overlay (1000 Hz + 1500 Hz)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    mask = demo_fft_mag > 1e-6
    axes[1, 1].stem(demo_freqs[mask], demo_fft_phase[mask], linefmt="C1-", markerfmt="C1o", basefmt=" ", label="NumPy FFT")
    axes[1, 1].stem(demo_freqs[mask], demo_dft_phase[mask], linefmt="C0--", markerfmt="C0x", basefmt=" ", label="Direct DFT")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Phase (degrees)")
    axes[1, 1].set_title("A4. Phase Accuracy on Active Tone Bins")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Group A: Direct DFT Mathematical Correctness vs NumPy FFT", fontsize=13, fontweight="bold")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "math_correctness.png", dpi=140)
    plt.close(fig)
    return rows


# ============================================================================
# Group B: Spectral Characteristics
# ============================================================================

def evaluate_spectral_characteristics():
    fs = 8000.0
    N = 160  # df = 50 Hz
    t = np.arange(N) / fs
    df = fs / N

    # Test fractional bin offsets from 0.0 (on-bin: 1000 Hz) to 0.5 (half-bin: 1025 Hz)
    offsets = [0.0, 0.1, 0.25, 0.4, 0.5]
    rows = []

    demo_on_rect = demo_off_rect = demo_on_hann = demo_off_hann = demo_freqs = None

    for frac in offsets:
        f_test = 1000.0 + frac * df
        sig = np.sin(2 * np.pi * f_test * t)

        for win_name in ("rect", "hann"):
            freqs, mag_dft, _ = sa.compute_spectrum(sig, fs, window=win_name, method="dft_vectorized")
            _, mag_fft, _ = sa.compute_spectrum(sig, fs, window=win_name, method="fft")
            pwr = mag_dft ** 2
            total_pwr = float(np.sum(pwr[1:]))
            idx_nearest = int(np.argmin(np.abs(freqs - f_test)))
            single_bin_pwr = float(pwr[idx_nearest])
            band3 = np.arange(max(1, idx_nearest - 1), min(len(pwr), idx_nearest + 2))
            band5 = np.arange(max(1, idx_nearest - 2), min(len(pwr), idx_nearest + 3))
            pwr_3bin = float(np.sum(pwr[band3]))
            pwr_5bin = float(np.sum(pwr[band5]))

            # Sidelobe peak outside +-2 bins from target
            outside_mask = np.ones(len(pwr), dtype=bool)
            outside_mask[0] = False
            outside_mask[max(1, idx_nearest - 2):min(len(pwr), idx_nearest + 3)] = False
            leakage_outside_ratio = float(np.sum(pwr[outside_mask]) / max(total_pwr, 1e-18))
            peak_sidelobe_db = float(
                10.0 * np.log10(max(float(np.max(pwr[outside_mask])), 1e-18) / max(single_bin_pwr, 1e-18))
            )

            det_f = sa.detect_frequency(sig, fs, window=win_name, method="dft_vectorized")
            rows.append({
                "window": win_name,
                "N": N,
                "fs_hz": fs,
                "freq_resolution_hz": df,
                "bin_offset_frac": frac,
                "true_freq_hz": f_test,
                "detected_bin_hz": det_f,
                "bin_freq_err_hz": abs(det_f - f_test),
                "single_bin_amp": f"{float(mag_dft[idx_nearest]):.6f}",
                "single_bin_power": f"{single_bin_pwr:.6f}",
                "integrated_3bin_power": f"{pwr_3bin:.6f}",
                "integrated_5bin_power": f"{pwr_5bin:.6f}",
                "leakage_outside_ratio": f"{leakage_outside_ratio:.6f}",
                "peak_sidelobe_db": f"{peak_sidelobe_db:.2f}",
                "dft_vs_fft_max_diff": f"{float(np.max(np.abs(mag_dft - mag_fft))):.3e}",
            })

            if frac == 0.0 and win_name == "rect":
                demo_freqs, demo_on_rect = freqs, mag_dft
            elif frac == 0.5 and win_name == "rect":
                demo_off_rect = mag_dft
            elif frac == 0.0 and win_name == "hann":
                demo_on_hann = mag_dft
            elif frac == 0.5 and win_name == "hann":
                demo_off_hann = mag_dft

    # Also test frequency resolution vs segment duration (two close tones 1000 Hz & 1050 Hz)
    res_rows = []
    for n_res in [80, 160, 320, 640]:
        t_r = np.arange(n_res) / fs
        sig_r = np.sin(2 * np.pi * 1000.0 * t_r) + np.sin(2 * np.pi * 1050.0 * t_r)
        freqs_r, mag_r, _ = sa.compute_spectrum(sig_r, fs, window="hann", method="dft_vectorized")
        res_rows.append((n_res, fs / n_res, freqs_r, mag_r))

    save_csv(OUT_DIR / "spectral_characteristics.csv", rows)

    # Plot Group B
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    zoom = (demo_freqs >= 600) & (demo_freqs <= 1400)
    axes[0, 0].plot(demo_freqs[zoom], 20 * np.log10(np.maximum(demo_on_rect[zoom], 1e-12)),
                    "o-", markersize=4, label="Rect (on-bin 1000 Hz)")
    axes[0, 0].plot(demo_freqs[zoom], 20 * np.log10(np.maximum(demo_off_rect[zoom], 1e-12)),
                    "s--", markersize=4, label="Rect (half-bin 1025 Hz)")
    axes[0, 0].plot(demo_freqs[zoom], 20 * np.log10(np.maximum(demo_off_hann[zoom], 1e-12)),
                    "^-", markersize=4, label="Hann (half-bin 1025 Hz)")
    axes[0, 0].set_xlabel("Frequency (Hz)")
    axes[0, 0].set_ylabel("Magnitude (dB)")
    axes[0, 0].set_title("B1. Spectral Leakage: Rectangular vs Hann Window")
    axes[0, 0].set_ylim(-90, 5)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    for win_name, style in (("rect", "o-"), ("hann", "s-")):
        sub = [r for r in rows if r["window"] == win_name]
        x_off = [r["bin_offset_frac"] for r in sub]
        leak = [float(r["leakage_outside_ratio"]) * 100.0 for r in sub]
        axes[0, 1].plot(x_off, leak, style, label=f"{win_name} outside +-2 bins (%)")
    axes[0, 1].set_xlabel("Fractional Bin Offset (bins)")
    axes[0, 1].set_ylabel("Out-of-Main-Lobe Leakage (%)")
    axes[0, 1].set_title("B2. Leakage Energy Across Off-Bin Frequencies")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    hann_sub = [r for r in rows if r["window"] == "hann"]
    x_off = [r["bin_offset_frac"] for r in hann_sub]
    axes[1, 0].plot(x_off, [float(r["single_bin_power"]) for r in hann_sub], "o-", label="Single bin (Hann)")
    axes[1, 0].plot(x_off, [float(r["integrated_3bin_power"]) for r in hann_sub], "s-", label="3-bin sum (Hann)")
    axes[1, 0].plot(x_off, [float(r["integrated_5bin_power"]) for r in hann_sub], "^-", label="5-bin sum (Hann)")
    axes[1, 0].set_xlabel("Fractional Bin Offset (bins)")
    axes[1, 0].set_ylabel("Captured Power")
    axes[1, 0].set_title("B3. Off-Bin Power Recovery (Single vs Multi-Bin Sum)")
    axes[1, 0].set_ylim(0, 1.2)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    for n_res, df_r, f_r, m_r in res_rows:
        z = (f_r >= 850) & (f_r <= 1200)
        axes[1, 1].plot(f_r[z], m_r[z], marker=".", label=f"N={n_res} (df={df_r:.1f} Hz)")
    axes[1, 1].axvline(1000.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1, 1].axvline(1050.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].set_title("B4. Frequency Resolution (1000 Hz + 1050 Hz, Hann)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Group B: Spectral Characteristics (Resolution, Leakage, Windowing)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "spectral_characteristics.png", dpi=140)
    plt.close(fig)
    return rows


# ============================================================================
# Group C: Practical Performance (Timing Scaling + Noise/BER + Threshold Checks)
# ============================================================================

def evaluate_practical_performance():
    rng = np.random.default_rng(SEED)
    rows = []

    # C1. Execution time scaling across lengths N
    lengths = [64, 128, 256, 512, 1024, 2048, 3528]
    timing_records = []
    for N in lengths:
        sig = rng.normal(0, 1, N)

        if N <= 512:
            reps_loop = 3 if N <= 256 else 1
            t0 = time.perf_counter()
            for _ in range(reps_loop):
                sa.direct_dft(sig)
            t_loop = (time.perf_counter() - t0) / reps_loop
        else:
            t_loop = None

        reps = 5
        t0 = time.perf_counter()
        for _ in range(reps):
            sa.direct_dft_vectorized(sig, block_size=256)
        t_vec = (time.perf_counter() - t0) / reps

        t0 = time.perf_counter()
        for _ in range(reps * 4):
            np.fft.fft(sig)
        t_fft = (time.perf_counter() - t0) / (reps * 4)

        timing_records.append((N, t_loop, t_vec, t_fft))
        rows.append({
            "experiment": "timing_scaling",
            "N_or_snr_db": N,
            "t_loop_s": f"{t_loop:.6f}" if t_loop is not None else "",
            "t_vec_dft_s": f"{t_vec:.6f}",
            "t_fft_s": f"{t_fft:.8f}",
            "speedup_vec_over_loop": f"{(t_loop / t_vec):.2f}" if t_loop is not None else "",
            "speedup_fft_over_vec": f"{(t_vec / max(t_fft, 1e-12)):.2f}",
            "legacy_ber": "",
            "dft_definite_ber": "",
            "dft_overall_err": "",
            "dft_detection_rate": "",
            "fft_definite_ber": "",
            "fft_detection_rate": "",
        })

    # C2. Noise sweep: BER & Detection Rate (Direct DFT vs FFT vs Legacy Un-gated)
    fs = 8000.0
    bit_dur = 0.05  # 400 samples/bit (fast, realistic df = 20 Hz)
    N_bit = int(round(bit_dur * fs))
    t_bit = np.arange(N_bit) / fs
    f0, f1 = 1000.0, 2000.0
    amp = 0.8
    tone_var = (amp ** 2) / 2.0
    n_trials = 120
    snr_db_levels = [10, 0, -6, -10, -14, -18, -22]
    ber_records = []

    for snr_db in snr_db_levels:
        noise_std = float(np.sqrt(tone_var / (10.0 ** (snr_db / 10.0))))
        bits_truth = rng.integers(0, 2, size=n_trials)

        legacy_err = 0
        dft_sub = dft_corr = dft_unc = 0
        fft_sub = fft_corr = fft_unc = 0

        for b_true in bits_truth:
            freq = f1 if b_true == 1 else f0
            clean = amp * np.sin(2 * np.pi * freq * t_bit)
            noisy = clean + rng.normal(0.0, noise_std, N_bit)

            # Legacy unwindowed single-bin comparison (always forces 0 or 1)
            X_leg = sa.direct_dft_vectorized(noisy)
            freqs_leg = sa.dft_frequencies(N_bit, fs)
            mag_leg = np.abs(X_leg[:len(freqs_leg)])
            p0_leg = mag_leg[int(np.argmin(np.abs(freqs_leg - f0)))] ** 2
            p1_leg = mag_leg[int(np.argmin(np.abs(freqs_leg - f1)))] ** 2
            leg_dec = 0 if p0_leg > p1_leg else 1
            if leg_dec != int(b_true):
                legacy_err += 1

            # Robust 4-outcome Direct DFT decoder
            d_out = sa.decode_bit(noisy, fs, f0=f0, f1=f1, method="dft_vectorized")
            if d_out in (0, 1):
                if d_out == int(b_true):
                    dft_corr += 1
                else:
                    dft_sub += 1
            else:
                dft_unc += 1

            # Identical 4-outcome gates via reference FFT
            f_out = sa.decode_bit(noisy, fs, f0=f0, f1=f1, method="fft")
            if f_out in (0, 1):
                if f_out == int(b_true):
                    fft_corr += 1
                else:
                    fft_sub += 1
            else:
                fft_unc += 1

        leg_ber = legacy_err / n_trials
        dft_decided = dft_corr + dft_sub
        dft_def_ber = (dft_sub / dft_decided) if dft_decided > 0 else 0.0
        dft_ovr_err = (dft_sub + dft_unc) / n_trials
        dft_det_rate = dft_decided / n_trials

        fft_decided = fft_corr + fft_sub
        fft_def_ber = (fft_sub / fft_decided) if fft_decided > 0 else 0.0
        fft_det_rate = fft_decided / n_trials

        ber_records.append((snr_db, leg_ber, dft_def_ber, dft_ovr_err, dft_det_rate, fft_def_ber, fft_det_rate))
        rows.append({
            "experiment": "snr_ber_sweep",
            "N_or_snr_db": snr_db,
            "t_loop_s": "",
            "t_vec_dft_s": "",
            "t_fft_s": "",
            "speedup_vec_over_loop": "",
            "speedup_fft_over_vec": "",
            "legacy_ber": f"{leg_ber:.4f}",
            "dft_definite_ber": f"{dft_def_ber:.4f}",
            "dft_overall_err": f"{dft_ovr_err:.4f}",
            "dft_detection_rate": f"{dft_det_rate:.4f}",
            "fft_definite_ber": f"{fft_def_ber:.4f}",
            "fft_detection_rate": f"{fft_det_rate:.4f}",
        })

    # C3. Experimental validation of silence, OOB, and ambiguous rejection
    special_cases = [
        ("silence", np.zeros(N_bit), "No signal"),
        ("out_of_band_440Hz", 0.8 * np.sin(2 * np.pi * 440.0 * t_bit), "?"),
        ("ambiguous_f0_plus_f1", 0.5 * np.sin(2 * np.pi * f0 * t_bit) + 0.5 * np.sin(2 * np.pi * f1 * t_bit), "?"),
        ("pure_white_noise", rng.normal(0, 0.4, N_bit), "?"),
    ]
    for cname, csig, expected in special_cases:
        info = sa.analyze_bit_interval(csig, fs, f0=f0, f1=f1, method="dft_vectorized")
        assert info["outcome"] == expected, f"Threshold check failed for {cname}: {info}"
        rows.append({
            "experiment": f"threshold_check_{cname}",
            "N_or_snr_db": info["reason"],
            "t_loop_s": "",
            "t_vec_dft_s": "",
            "t_fft_s": "",
            "speedup_vec_over_loop": "",
            "speedup_fft_over_vec": "",
            "legacy_ber": "",
            "dft_definite_ber": "",
            "dft_overall_err": "",
            "dft_detection_rate": 0.0,
            "fft_definite_ber": "",
            "fft_detection_rate": "",
        })

    save_csv(OUT_DIR / "practical_performance.csv", rows)

    # Plot Group C
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ns = [r[0] for r in timing_records]
    loop_ns = [r[0] for r in timing_records if r[1] is not None]
    loop_ts = [r[1] for r in timing_records if r[1] is not None]
    vec_ts = [r[2] for r in timing_records]
    fft_ts = [r[3] for r in timing_records]

    axes[0].loglog(loop_ns, loop_ts, "o-", label="direct_dft (nested loop)")
    axes[0].loglog(ns, vec_ts, "s-", label="direct_dft_vectorized (blocks)")
    axes[0].loglog(ns, fft_ts, "^-", label="np.fft.fft (reference)")
    axes[0].set_xlabel("Signal Length N (samples)")
    axes[0].set_ylabel("Execution Time (s)")
    axes[0].set_title("C1. Runtime Scaling vs Signal Length")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)

    snrs = [r[0] for r in ber_records]
    axes[1].plot(snrs, [r[1] for r in ber_records], "o--", label="Legacy un-gated BER")
    axes[1].plot(snrs, [r[2] for r in ber_records], "s-", label="Direct DFT Definite BER")
    axes[1].plot(snrs, [r[5] for r in ber_records], "^:", label="Reference FFT Definite BER")
    axes[1].set_xlabel("Input Tone SNR (dB)")
    axes[1].set_ylabel("Bit Error Rate (BER)")
    axes[1].set_title("C2. Definite BER vs Input SNR")
    axes[1].set_ylim(-0.03, 0.65)
    axes[1].invert_xaxis()
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(snrs, [r[4] * 100.0 for r in ber_records], "s-", label="Direct DFT Detection Rate (%)")
    axes[2].plot(snrs, [r[6] * 100.0 for r in ber_records], "^:", label="Reference FFT Detection Rate (%)")
    axes[2].plot(snrs, [r[3] * 100.0 for r in ber_records], "o--", label="Direct DFT Overall Error+Erasure (%)")
    axes[2].set_xlabel("Input Tone SNR (dB)")
    axes[2].set_ylabel("Percentage (%)")
    axes[2].set_title("C3. Detection Rate vs Unreliable Rate")
    axes[2].set_ylim(-5, 105)
    axes[2].invert_xaxis()
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Group C: Practical Performance (Execution Time, BER, Detection Rate)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "practical_performance.png", dpi=140)
    plt.close(fig)
    return timing_records, ber_records


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Running Group A: Mathematical Correctness...")
    math_rows = evaluate_mathematical_correctness()
    for r in math_rows:
        print(f"  {r['case']:<28} max_err_loop={r['max_abs_err_loop']} max_err_vec={r['max_abs_err_vec']}")

    print("\nRunning Group B: Spectral Characteristics...")
    spec_rows = evaluate_spectral_characteristics()
    for r in spec_rows:
        if r["bin_offset_frac"] in (0.0, 0.5):
            print(f"  win={r['window']:<4} offset={r['bin_offset_frac']:.1f} "
                  f"leak={r['leakage_outside_ratio']} sidelobe_dB={r['peak_sidelobe_db']} "
                  f"1binP={r['single_bin_power']} 3binP={r['integrated_3bin_power']}")

    print("\nRunning Group C: Practical Performance...")
    timing_records, ber_records = evaluate_practical_performance()
    for N, tl, tv, tf in timing_records:
        tl_str = f"{tl:.5f}s" if tl is not None else "skipped"
        print(f"  N={N:<4} loop={tl_str:<10} vec_dft={tv:.5f}s  fft={tf:.6f}s")
    print("\n  SNR(dB) | Legacy BER | DFT Def BER | DFT Det Rate | FFT Def BER")
    for snr_db, leg, d_ber, d_ovr, d_det, f_ber, f_det in ber_records:
        print(f"  {snr_db:>6}  | {leg:>10.4f} | {d_ber:>11.4f} | {d_det:>11.2%} | {f_ber:>11.4f}")

    print(f"\nSaved all CSVs and plots under: {OUT_DIR}")


if __name__ == "__main__":
    main()
