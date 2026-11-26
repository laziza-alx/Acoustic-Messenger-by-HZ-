"""
signal_analysis.py -- Direct DFT and Robust FSK Signal Analysis Core

Provides:
1. Reference nested-loop Direct DFT (`direct_dft`).
2. Memory-bounded block-vectorized Direct DFT (`direct_dft_vectorized`) and
   selective-bin Direct DFT (`direct_dft_bins`), without using FFT.
3. Standalone WAV loader and windowed spectral analysis (`load_wav`,
   `compute_spectrum`).
4. Four-outcome FSK bit decoder (`0`, `1`, `?`, `No signal`) with noise-floor/SNR,
   silence, contrast, and out-of-band interference checks.
5. Independent FSK message boundary detector (`find_message_bounds`,
   `refine_start`) that does not use the ground-truth bit sequence.
6. Explicit bit-sequence comparison and BER/erasure metrics
   (`compare_bit_sequences`).
"""

import wave
import numpy as np

try:
    from scipy.io import wavfile as _scipy_wavfile
except ImportError:
    _scipy_wavfile = None


DEFAULT_GUARD = 0.1
DEFAULT_BAND_BINS = 2
DEFAULT_MIN_POWER = 1e-6
DEFAULT_MIN_SNR = 5.0
DEFAULT_MIN_BAND_RATIO = 0.25
DEFAULT_MIN_CONTRAST = 0.25


# ============================================================================
# 1. Direct Discrete Fourier Transform implementations (no FFT)
# ============================================================================

def direct_dft(signal):
    """Reference O(N^2) Direct DFT using explicit nested loops."""
    x = np.asarray(signal, dtype=complex)
    N = len(x)
    result = np.zeros(N, dtype=complex)

    for k in range(N):
        for n in range(N):
            angle = -2 * np.pi * k * n / N
            result[k] += x[n] * (np.cos(angle) + 1j * np.sin(angle))

    return result


def direct_dft_vectorized(signal, block_size=256):
    """Exact O(N^2) Direct DFT via NumPy row-blocks (avoids N x N allocation).

    Computes X[k] = sum_{n=0}^{N-1} x[n] * exp(-j * 2 * pi * k * n / N)
    in chunks of `block_size` frequency bins so peak memory is O(block_size * N).
    For real inputs, exploits Hermitian symmetry X[N - k] = conj(X[k]) to halve
    the number of computed bins.
    """
    x = np.asarray(signal)
    N = len(x)
    if N == 0:
        return np.zeros(0, dtype=np.complex128)

    is_real = np.isrealobj(x)
    x_c = np.asarray(x, dtype=np.complex128)
    n = np.arange(N, dtype=np.float64)
    result = np.empty(N, dtype=np.complex128)
    step = max(1, int(block_size))
    limit = (N // 2 + 1) if is_real else N

    for k_start in range(0, limit, step):
        k_end = min(k_start + step, limit)
        k = np.arange(k_start, k_end, dtype=np.float64)[:, None]
        basis = np.exp(-2j * np.pi * (k * n[None, :]) / N)
        result[k_start:k_end] = basis @ x_c

    if is_real and N > 2:
        # Fill negative-frequency half via X[N - k] = conj(X[k]).
        mirror_end = (N + 1) // 2
        result[N // 2 + 1:] = np.conj(result[1:mirror_end][::-1])

    return result


def direct_dft_batch(signals, fs, window="hann", block_size=256):
    """Compute one-sided Direct DFT spectra for a 2D batch (M, N) of equal-length
    real bit segments, reusing each block_size x N DFT basis across all M segments."""
    arr = np.asarray(signals, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return np.zeros(0), np.zeros((0, 0)), np.zeros((0, 0), dtype=np.complex128)

    M, N = arr.shape
    win = make_window(N, window)
    win_sum = float(np.sum(win))
    if win_sum <= 0:
        win = np.ones(N, dtype=np.float64)
        win_sum = float(N)

    weighted = (arr * win[None, :]).astype(np.complex128)
    half = N // 2 + 1
    n = np.arange(N, dtype=np.float64)
    step = max(1, int(block_size))
    X_half = np.empty((M, half), dtype=np.complex128)

    for k_start in range(0, half, step):
        k_end = min(k_start + step, half)
        k = np.arange(k_start, k_end, dtype=np.float64)[:, None]
        basis = np.exp(-2j * np.pi * (k * n[None, :]) / N)
        X_half[:, k_start:k_end] = weighted @ basis.T

    freqs = dft_frequencies(N, fs)
    mags = np.abs(X_half) * 2.0 / win_sum
    mags[:, 0] *= 0.5
    if N % 2 == 0 and half > 1:
        mags[:, -1] *= 0.5
    return freqs, mags, X_half


def direct_dft_bins(signal, bin_indices):
    """Exact Direct DFT evaluated only at the requested integer bin indices."""
    x = np.asarray(signal, dtype=np.complex128)
    N = len(x)
    k = np.asarray(bin_indices, dtype=np.float64)
    if N == 0 or len(k) == 0:
        return np.zeros(len(k), dtype=np.complex128)

    n = np.arange(N, dtype=np.float64)
    basis = np.exp(-2j * np.pi * (k[:, None] * n[None, :]) / N)
    return basis @ x


def dft_frequencies(N, fs):
    """Return one-sided frequency bin centers in Hz for length N at fs Hz."""
    if N <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.arange(N // 2 + 1, dtype=np.float64) * (float(fs) / N)


# ============================================================================
# 2. WAV loading and windowed spectrum computation
# ============================================================================

def _read_wav_stdlib(path):
    """Fallback PCM WAV reader using Python's built-in wave module."""
    with wave.open(str(path), "rb") as w:
        fs = w.getframerate()
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)

    if sampwidth == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (b[:, 0].astype(np.int32)
                  | (b[:, 1].astype(np.int32) << 8)
                  | (b[:, 2].astype(np.int32) << 16))
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        x = signed.astype(np.float64) / float(1 << 23)
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)
    return int(fs), x


def load_wav(path):
    """Read a WAV file, normalize to [-1, 1] mono float64, and remove DC."""
    if _scipy_wavfile is not None:
        fs, x = _scipy_wavfile.read(path)
        if x.dtype == np.uint8:
            x = (x.astype(np.float64) - 128.0) / 128.0
        elif np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
        else:
            x = x.astype(np.float64)

        if x.ndim == 2:
            x = x.mean(axis=1)
    else:
        fs, x = _read_wav_stdlib(path)

    if len(x) > 0:
        x = x - np.mean(x)

    return int(fs), x


def make_window(N, window_type="hann"):
    """Return window coefficients for length N ('hann' or 'rect')."""
    if N <= 0:
        return np.zeros(0, dtype=np.float64)
    kind = (window_type or "hann").strip().lower()
    if kind in ("hann", "hanning"):
        return np.hanning(N)
    if kind in ("rect", "rectangular", "boxcar", "none"):
        return np.ones(N, dtype=np.float64)
    raise ValueError(f"Unsupported window type: {window_type}")


def compute_spectrum(segment, fs, window="hann", method="dft_vectorized", block_size=256):
    """Compute normalized one-sided spectrum using Direct DFT or reference FFT.

    Returns (frequencies_hz, one_sided_amplitude, full_complex_spectrum).
    """
    x = np.asarray(segment, dtype=np.float64)
    N = len(x)
    if N == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, np.zeros(0, dtype=np.complex128)

    win = make_window(N, window)
    win_sum = float(np.sum(win))
    if win_sum <= 0:
        win = np.ones(N, dtype=np.float64)
        win_sum = float(N)

    weighted = x * win
    m = (method or "dft_vectorized").strip().lower()
    if m in ("dft_vectorized", "dft", "vectorized"):
        X = direct_dft_vectorized(weighted, block_size=block_size)
    elif m in ("dft_loop", "loop"):
        X = direct_dft(weighted)
    elif m == "fft":
        X = np.fft.fft(weighted)
    else:
        raise ValueError(f"Unsupported spectrum method: {method}")

    freqs = dft_frequencies(N, fs)
    half = N // 2 + 1
    mag = np.abs(X[:half]) * 2.0 / win_sum
    mag[0] *= 0.5
    if N % 2 == 0 and half > 1:
        mag[-1] *= 0.5

    return freqs, mag, X


# ============================================================================
# 3. Single-segment frequency detection & robust 4-outcome FSK decoding
# ============================================================================

def detect_frequency(signal, fs, window="hann", method="dft_vectorized"):
    """Return the strongest non-DC frequency (Hz) in `signal`."""
    freqs, mag, _ = compute_spectrum(signal, fs, window=window, method=method)
    if len(mag) <= 1:
        return 0.0
    mag_no_dc = mag.copy()
    mag_no_dc[0] = 0.0
    return float(freqs[int(np.argmax(mag_no_dc))])


def analyze_signal(signal, fs, window="hann", method="dft_vectorized"):
    """Return (frequencies_hz, amplitude_spectrum, phase_deg) for positive bins."""
    freqs, mag, X = compute_spectrum(signal, fs, window=window, method=method)
    half = len(freqs)
    phase = np.angle(X[:half], deg=True) if half > 0 else np.zeros(0)
    return freqs, mag, phase


def _target_band_indices(freqs, f0, f1, band_bins=DEFAULT_BAND_BINS):
    """Return disjoint bin index slices around f0 and f1."""
    n_bins = len(freqs)
    if n_bins <= 1:
        return np.array([], dtype=int), np.array([], dtype=int)

    df = float(freqs[1] - freqs[0]) if n_bins > 1 else 1.0
    idx0 = int(np.argmin(np.abs(freqs - f0)))
    idx1 = int(np.argmin(np.abs(freqs - f1)))

    # Keep f0 and f1 integration bands strictly disjoint.
    max_half_bins = max(0, int(np.floor(abs(f1 - f0) / (3.0 * max(df, 1e-12)))))
    half_bins = min(max(0, int(band_bins)), max_half_bins)

    b0 = np.arange(max(1, idx0 - half_bins), min(n_bins, idx0 + half_bins + 1))
    b1 = np.arange(max(1, idx1 - half_bins), min(n_bins, idx1 + half_bins + 1))
    return b0, b1


def _evaluate_spectrum_gates(
    freqs,
    mag,
    f0=1000.0,
    f1=2000.0,
    band_bins=DEFAULT_BAND_BINS,
    min_power=DEFAULT_MIN_POWER,
    min_snr=DEFAULT_MIN_SNR,
    min_band_ratio=DEFAULT_MIN_BAND_RATIO,
    min_contrast=DEFAULT_MIN_CONTRAST,
):
    """Apply the 4 decision gates on a precomputed one-sided amplitude spectrum."""
    if len(freqs) <= 2:
        return {
            "outcome": "No signal",
            "symbol": "-",
            "reason": "segment_too_short",
            "power_0": 0.0,
            "power_1": 0.0,
            "mag_0": 0.0,
            "mag_1": 0.0,
            "noise_floor": 0.0,
            "snr": 0.0,
            "in_band_ratio": 0.0,
            "contrast": 0.0,
            "peak_freq_hz": 0.0,
            "freqs": freqs,
            "magnitude": mag,
        }

    power = mag ** 2
    b0, b1 = _target_band_indices(freqs, f0, f1, band_bins=band_bins)

    power_0 = float(np.sum(power[b0])) if len(b0) else 0.0
    power_1 = float(np.sum(power[b1])) if len(b1) else 0.0
    mag_0 = float(np.sqrt(power_0))
    mag_1 = float(np.sqrt(power_1))
    total_power = float(np.sum(power[1:]))

    half_guard = max(2, (len(b0) // 2) + 2)
    idx0 = int(np.argmin(np.abs(freqs - f0)))
    idx1 = int(np.argmin(np.abs(freqs - f1)))
    noise_mask = np.ones(len(power), dtype=bool)
    noise_mask[0] = False
    noise_mask[max(1, idx0 - half_guard):min(len(power), idx0 + half_guard + 1)] = False
    noise_mask[max(1, idx1 - half_guard):min(len(power), idx1 + half_guard + 1)] = False

    if np.any(noise_mask):
        noise_per_bin = float(np.median(power[noise_mask]) / np.log(2.0))
    else:
        noise_per_bin = 0.0

    n_band = max(len(b0), len(b1), 1)
    expected_band_noise = noise_per_bin * n_band
    max_target_power = max(power_0, power_1)

    snr = max_target_power / max(expected_band_noise, 1e-18)
    in_band_ratio = (power_0 + power_1) / max(total_power, 1e-18)
    contrast = abs(power_0 - power_1) / max(power_0 + power_1, 1e-18)
    peak_idx = int(np.argmax(power[1:])) + 1
    peak_freq_hz = float(freqs[peak_idx])

    if total_power < min_power and max_target_power < min_power:
        outcome, symbol, reason = "No signal", "-", "silence"
    elif max_target_power < min_power:
        outcome, symbol, reason = "?", "?", "out_of_band_only"
    elif expected_band_noise > 0.0 and snr < min_snr:
        outcome, symbol, reason = "?", "?", "low_snr"
    elif in_band_ratio < min_band_ratio:
        outcome, symbol, reason = "?", "?", "out_of_band_interference"
    elif contrast < min_contrast:
        outcome, symbol, reason = "?", "?", "low_contrast"
    elif power_0 > power_1:
        outcome, symbol, reason = "0", "0", "ok"
    else:
        outcome, symbol, reason = "1", "1", "ok"

    return {
        "outcome": outcome,
        "symbol": symbol,
        "reason": reason,
        "power_0": power_0,
        "power_1": power_1,
        "mag_0": mag_0,
        "mag_1": mag_1,
        "noise_floor": expected_band_noise,
        "snr": float(snr),
        "in_band_ratio": float(in_band_ratio),
        "contrast": float(contrast),
        "peak_freq_hz": peak_freq_hz,
        "freqs": freqs,
        "magnitude": mag,
    }


def analyze_bit_interval(
    signal,
    fs,
    f0=1000.0,
    f1=2000.0,
    window="hann",
    method="dft_vectorized",
    band_bins=DEFAULT_BAND_BINS,
    min_power=DEFAULT_MIN_POWER,
    min_snr=DEFAULT_MIN_SNR,
    min_band_ratio=DEFAULT_MIN_BAND_RATIO,
    min_contrast=DEFAULT_MIN_CONTRAST,
):
    """Evaluate one bit interval and return 4-outcome decision plus diagnostics."""
    freqs, mag, _ = compute_spectrum(signal, fs, window=window, method=method)
    return _evaluate_spectrum_gates(
        freqs,
        mag,
        f0=f0,
        f1=f1,
        band_bins=band_bins,
        min_power=min_power,
        min_snr=min_snr,
        min_band_ratio=min_band_ratio,
        min_contrast=min_contrast,
    )


def decode_bit(
    signal,
    fs,
    f0=1000.0,
    f1=2000.0,
    window="hann",
    method="dft_vectorized",
    band_bins=DEFAULT_BAND_BINS,
    min_power=DEFAULT_MIN_POWER,
    min_snr=DEFAULT_MIN_SNR,
    min_band_ratio=DEFAULT_MIN_BAND_RATIO,
    min_contrast=DEFAULT_MIN_CONTRAST,
):
    """Decode a single bit segment into 0, 1, '?', or 'No signal'."""
    info = analyze_bit_interval(
        signal,
        fs,
        f0=f0,
        f1=f1,
        window=window,
        method=method,
        band_bins=band_bins,
        min_power=min_power,
        min_snr=min_snr,
        min_band_ratio=min_band_ratio,
        min_contrast=min_contrast,
    )
    if info["outcome"] == "0":
        return 0
    if info["outcome"] == "1":
        return 1
    return info["outcome"]


# ============================================================================
# 4. Independent WAV message boundary detection & multi-bit decoding
# ============================================================================

def _sliding_fsk_envelope(x, fs, f0, f1, frame_s, hop_s):
    """Evaluate FSK in-band and background noise power across sliding frames
    using selective-bin Direct DFT (without FFT or N x N allocation)."""
    N_frame = max(16, int(round(frame_s * fs)))
    N_hop = max(1, int(round(hop_s * fs)))
    if len(x) < N_frame:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    win = np.hanning(N_frame)
    win_sum = float(np.sum(win))
    freqs = dft_frequencies(N_frame, fs)

    idx0 = int(np.argmin(np.abs(freqs - f0)))
    idx1 = int(np.argmin(np.abs(freqs - f1)))
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    max_half = max(0, int(np.floor(abs(f1 - f0) / (3.0 * max(df, 1e-12)))))
    half = min(1, max_half)

    b0_bins = np.arange(max(1, idx0 - half), min(len(freqs), idx0 + half + 1))
    b1_bins = np.arange(max(1, idx1 - half), min(len(freqs), idx1 + half + 1))
    fsk_bins = np.unique(np.concatenate([b0_bins, b1_bins]))

    # Sample ~12 out-of-band bins across [150 Hz, 0.9 * fs/2] for noise floor.
    candidate_bins = np.arange(max(1, int(round(150.0 / df))), max(2, len(freqs) - 2))
    exclude = set(range(max(1, idx0 - half - 2), min(len(freqs), idx0 + half + 3))) | set(
        range(max(1, idx1 - half - 2), min(len(freqs), idx1 + half + 3))
    )
    bg_candidates = np.array([b for b in candidate_bins if b not in exclude], dtype=int)
    if len(bg_candidates) > 12:
        bg_bins = bg_candidates[np.linspace(0, len(bg_candidates) - 1, 12, dtype=int)]
    else:
        bg_bins = bg_candidates

    all_bins = np.unique(np.concatenate([fsk_bins, bg_bins]))
    fsk_mask = np.isin(all_bins, fsk_bins)
    bg_mask = np.isin(all_bins, bg_bins)

    n = np.arange(N_frame, dtype=np.float64)
    basis = np.exp(-2j * np.pi * (all_bins.astype(np.float64)[:, None] * n[None, :]) / N_frame)

    starts = np.arange(0, len(x) - N_frame + 1, N_hop)
    n_frames = len(starts)
    fsk_power = np.zeros(n_frames, dtype=np.float64)
    bg_band_noise = np.zeros(n_frames, dtype=np.float64)
    scale = 2.0 / win_sum
    n_fsk_half = max(len(b0_bins), len(b1_bins), 1)

    batch_size = 512
    for b_start in range(0, n_frames, batch_size):
        b_end = min(b_start + batch_size, n_frames)
        idx_mat = starts[b_start:b_end, None] + np.arange(N_frame)[None, :]
        seg_batch = x[idx_mat] * win[None, :]
        spec = np.abs(seg_batch @ basis.T) * scale
        pwr = spec ** 2
        fsk_power[b_start:b_end] = np.sum(pwr[:, fsk_mask], axis=1)
        if np.any(bg_mask):
            bg_per_bin = np.median(pwr[:, bg_mask], axis=1) / np.log(2.0)
            bg_band_noise[b_start:b_end] = bg_per_bin * n_fsk_half

    times = (starts + 0.5 * N_frame) / float(fs)
    return times, fsk_power, bg_band_noise


def _close_and_clean_mask(mask, min_gap_frames, min_run_frames):
    """Drop short isolated spikes first, then bridge brief internal dropouts."""
    m = mask.copy()
    n = len(m)
    if n == 0:
        return m

    # 1. Remove True runs shorter than min_run_frames so noise glitches aren't bridged.
    if min_run_frames > 1:
        i = 0
        while i < n:
            if m[i]:
                j = i
                while j < n and m[j]:
                    j += 1
                if (j - i) < min_run_frames:
                    m[i:j] = False
                i = j
            else:
                i += 1

    # 2. Bridge False runs shorter than min_gap_frames strictly between True regions.
    if min_gap_frames > 0:
        i = 0
        while i < n:
            if not m[i]:
                j = i
                while j < n and not m[j]:
                    j += 1
                if i > 0 and j < n and (j - i) <= min_gap_frames:
                    m[i:j] = True
                i = j
            else:
                i += 1

    return m


def find_message_bounds(
    x,
    fs,
    f0,
    f1,
    bit_duration,
    min_power=DEFAULT_MIN_POWER,
    active_snr=5.0,
    frame_s=None,
    hop_s=None,
):
    """Detect message (start_s, end_s, found) independently of original bits."""
    total_dur = len(x) / float(fs) if fs > 0 else 0.0
    if len(x) == 0 or total_dur < 0.4 * bit_duration:
        return 0.0, 0.0, False

    if frame_s is None:
        frame_s = min(0.025, max(0.008, 0.25 * bit_duration))
    if hop_s is None:
        hop_s = min(0.002, max(0.0005, 0.05 * bit_duration))

    times, fsk_power, bg_noise = _sliding_fsk_envelope(x, fs, f0, f1, frame_s, hop_s)
    if len(times) == 0:
        return 0.0, 0.0, False

    smooth_frames = max(1, int(round(min(0.02, 0.25 * bit_duration) / hop_s)))
    kernel = np.ones(smooth_frames, dtype=np.float64) / smooth_frames
    fsk_smooth = np.convolve(fsk_power, kernel, mode="same")
    noise_smooth = np.convolve(bg_noise, kernel, mode="same")

    peak_power = float(np.max(fsk_smooth))
    if peak_power < min_power:
        return 0.0, 0.0, False

    peak_db = 10.0 * np.log10(peak_power + 1e-18)
    fsk_db = 10.0 * np.log10(fsk_smooth + 1e-18)
    snr_ratio = fsk_smooth / np.maximum(noise_smooth, 1e-18)

    # Frame is active if it passes absolute silence floor, dynamic-range floor,
    # and in-frame SNR over out-of-band noise.
    active = (
        (fsk_smooth >= min_power)
        & (fsk_db >= peak_db - 18.0)
        & ((noise_smooth <= 1e-15) | (snr_ratio >= active_snr))
    )

    min_gap = max(1, int(round(2.5 * bit_duration / hop_s)))
    min_run = max(1, int(round(0.5 * bit_duration / hop_s)))
    active = _close_and_clean_mask(active, min_gap, min_run)

    if not np.any(active):
        return 0.0, 0.0, False

    # Select the longest contiguous active region (ties broken by highest energy).
    best_span = None
    best_score = -1.0
    i = 0
    n = len(active)
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            score = float(np.sum(fsk_smooth[i:j]))
            if score > best_score:
                best_score = score
                best_span = (i, j - 1)
            i = j
        else:
            i += 1

    start_idx, end_idx = best_span
    half_frame = 0.5 * frame_s
    t_start = max(0.0, float(times[start_idx] - half_frame))
    t_end = min(total_dur, float(times[end_idx] + half_frame))
    return t_start, t_end, True


def refine_start(x, fs, t_coarse_start, t_coarse_end, bit_duration, f0, f1, guard=DEFAULT_GUARD, step=0.001):
    """Fine-tune message start near t_coarse_start by maximizing full-bit FSK separation."""
    est_bits = max(1, int(round((t_coarse_end - t_coarse_start) / bit_duration)))
    eval_bits = min(est_bits, 32)
    candidates = np.arange(
        max(0.0, t_coarse_start - 0.5 * bit_duration),
        min(len(x) / float(fs) - bit_duration, t_coarse_start + 0.5 * bit_duration) + 1e-9,
        step,
    )
    if len(candidates) == 0:
        return t_coarse_start

    # Use full bit intervals (no guard) so misaligning into adjacent bits or silence
    # immediately lowers |P1 - P0|, yielding a sharp peak at the true bit boundary.
    seg_len = max(16, int(round(bit_duration * fs)))
    win = np.hanning(seg_len)
    scale = 2.0 / max(float(np.sum(win)), 1e-12)
    freqs = dft_frequencies(seg_len, fs)
    b0, b1 = _target_band_indices(freqs, f0, f1, band_bins=1)
    bins = np.unique(np.concatenate([b0, b1]))
    is_b0 = np.isin(bins, b0)
    is_b1 = np.isin(bins, b1)
    n = np.arange(seg_len, dtype=np.float64)
    basis = np.exp(-2j * np.pi * (bins.astype(np.float64)[:, None] * n[None, :]) / seg_len)

    best_t = t_coarse_start
    best_score = -1.0

    for cand in candidates:
        scores = []
        for i in range(eval_bits):
            a = int(round((cand + i * bit_duration) * fs))
            b = a + seg_len
            if a < 0 or b > len(x):
                continue
            spec = np.abs(basis @ (x[a:b] * win)) * scale
            p0 = float(np.sum(spec[is_b0] ** 2))
            p1 = float(np.sum(spec[is_b1] ** 2))
            scores.append(abs(p1 - p0))
        if scores:
            score = float(np.mean(scores))
            if score > best_score:
                best_score = score
                best_t = float(cand)

    return best_t


def decode_signal_message(
    x,
    fs,
    f0=1000.0,
    f1=2000.0,
    bit_duration=0.1,
    guard=DEFAULT_GUARD,
    offset_ms=0.0,
    window="hann",
    method="dft_vectorized",
    band_bins=DEFAULT_BAND_BINS,
    min_power=DEFAULT_MIN_POWER,
    min_snr=DEFAULT_MIN_SNR,
    min_band_ratio=DEFAULT_MIN_BAND_RATIO,
    min_contrast=DEFAULT_MIN_CONTRAST,
    active_snr=5.0,
):
    """Independently detect boundaries and decode all bits in a WAV signal."""
    t_coarse_start, t_coarse_end, found = find_message_bounds(
        x, fs, f0, f1, bit_duration, min_power=min_power, active_snr=active_snr
    )
    if not found:
        return {
            "found": False,
            "t_coarse_start": 0.0,
            "t_coarse_end": 0.0,
            "t_start": 0.0,
            "t_end": 0.0,
            "n_bits_detected": 0,
            "decoded_sequence": "",
            "bit_results": [],
        }

    t_refined = refine_start(
        x, fs, t_coarse_start, t_coarse_end, bit_duration, f0, f1, guard=guard
    )
    t_start = max(0.0, t_refined + offset_ms / 1000.0)

    active_span = max(0.0, t_coarse_end - t_start)
    max_possible = max(0, int(np.floor((len(x) / float(fs) - t_start + 0.15 * bit_duration) / bit_duration)))
    n_bits = min(max_possible, max(1, int(round(active_span / bit_duration))))

    seg_len = int(round((1.0 - 2.0 * guard) * bit_duration * fs))
    slices = []
    valid_indices = []
    for i in range(n_bits):
        t_bit_start = t_start + i * bit_duration
        t_from = t_bit_start + guard * bit_duration
        a = max(0, int(round(t_from * fs)))
        b = a + seg_len
        if seg_len >= 16 and b <= len(x):
            slices.append((i, t_bit_start, t_from, t_from + seg_len / float(fs), x[a:b]))
            valid_indices.append(i)
        else:
            b_clip = min(len(x), int(round((t_bit_start + (1.0 - guard) * bit_duration) * fs)))
            slices.append((i, t_bit_start, t_from, b_clip / float(fs), x[a:b_clip]))

    # Batch equal-length bit intervals when using vectorized Direct DFT so basis blocks are computed once.
    batch_spectra = {}
    if method in ("dft_vectorized", "dft", "vectorized") and len(valid_indices) > 1:
        batch_arr = np.stack([slices[idx][4] for idx in valid_indices], axis=0)
        batch_freqs, batch_mags, _ = direct_dft_batch(batch_arr, fs, window=window)
        for pos, idx in enumerate(valid_indices):
            batch_spectra[idx] = (batch_freqs, batch_mags[pos])

    bit_results = []
    symbols = []
    for i, t_bit_start, t_from, t_to, seg in slices:
        if len(seg) < 16:
            info = _evaluate_spectrum_gates(np.zeros(0), np.zeros(0))
            info["reason"] = "truncated_interval"
        elif i in batch_spectra:
            bfreqs, bmag = batch_spectra[i]
            info = _evaluate_spectrum_gates(
                bfreqs,
                bmag,
                f0=f0,
                f1=f1,
                band_bins=band_bins,
                min_power=min_power,
                min_snr=min_snr,
                min_band_ratio=min_band_ratio,
                min_contrast=min_contrast,
            )
        else:
            info = analyze_bit_interval(
                seg,
                fs,
                f0=f0,
                f1=f1,
                window=window,
                method=method,
                band_bins=band_bins,
                min_power=min_power,
                min_snr=min_snr,
                min_band_ratio=min_band_ratio,
                min_contrast=min_contrast,
            )

        info["bit_index"] = i
        info["t_bit_start"] = t_bit_start
        info["t_from"] = t_from
        info["t_to"] = t_to
        bit_results.append(info)
        symbols.append(info["symbol"])

    return {
        "found": True,
        "t_coarse_start": t_coarse_start,
        "t_coarse_end": t_coarse_end,
        "t_start": t_start,
        "t_end": t_start + n_bits * bit_duration,
        "n_bits_detected": n_bits,
        "decoded_sequence": "".join(symbols),
        "bit_results": bit_results,
    }


# ============================================================================
# 5. Bit-by-bit comparison and BER / erasure reporting
# ============================================================================

def compare_bit_sequences(decoded_sequence, original_sequence):
    """Compare independently decoded bits against ground-truth bits.

    Symbols in `decoded_sequence`:
    - '0', '1' : decided bits
    - '?'      : uncertain / ambiguous bit (low SNR, contrast, or out-of-band)
    - '-'      : 'No signal' (silence / missing interval energy)

    Formulas (L_orig = len(original), L_dec = len(decoded), L_cmp = min(L_orig, L_dec)):
    - correct_bits (C): aligned positions i < L_cmp where dec[i] == orig[i]
    - substitutions (S): aligned positions i < L_cmp where dec[i] in ('0','1') and dec[i] != orig[i]
    - uncertain_bits (U): aligned positions i < L_cmp where dec[i] == '?'
    - no_signal_bits (Ns): aligned positions i < L_cmp where dec[i] == '-'
    - missing_bits (M): max(0, L_orig - L_dec)
    - extra_bits (X): max(0, L_dec - L_orig)
    - definite_ber = S / (C + S)   (NaN if C + S == 0)
    - overall_error_rate = (S + U + Ns + M + X) / L_orig
    """
    orig = (original_sequence or "").strip()
    dec = (decoded_sequence or "").strip()
    L_orig = len(orig)
    L_dec = len(dec)
    L_cmp = min(L_orig, L_dec)

    correct = 0
    substitutions = 0
    uncertain = 0
    no_signal = 0
    per_position = []

    for i in range(max(L_orig, L_dec)):
        o_ch = orig[i] if i < L_orig else "<extra>"
        d_ch = dec[i] if i < L_dec else "<missing>"
        if i < L_cmp:
            if d_ch in ("0", "1"):
                if d_ch == o_ch:
                    correct += 1
                    status = "correct"
                else:
                    substitutions += 1
                    status = "substitution"
            elif d_ch == "?":
                uncertain += 1
                status = "uncertain"
            else:
                no_signal += 1
                status = "no_signal"
        elif i >= L_dec:
            status = "missing"
        else:
            status = "extra"
        per_position.append({"index": i, "original": o_ch, "decoded": d_ch, "status": status})

    missing = max(0, L_orig - L_dec)
    extra = max(0, L_dec - L_orig)
    erasures = uncertain + no_signal
    decided = correct + substitutions

    definite_ber = (substitutions / decided) if decided > 0 else float("nan")
    overall_error_rate = ((substitutions + erasures + missing + extra) / L_orig) if L_orig > 0 else float("nan")

    return {
        "total_original_bits": L_orig,
        "total_decoded_bits": L_dec,
        "correct_bits": correct,
        "substitutions": substitutions,
        "uncertain_bits": uncertain,
        "no_signal_bits": no_signal,
        "erasures": erasures,
        "missing_bits": missing,
        "extra_bits": extra,
        "decided_bits": decided,
        "definite_ber": definite_ber,
        "overall_error_rate": overall_error_rate,
        "per_position": per_position,
    }
