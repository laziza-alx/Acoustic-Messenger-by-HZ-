"""Plot BER summaries from the existing Acoustic Messenger receiver's trials.csv.

Usage (from the project root):
    python plot_results.py
    python plot_results.py --csv results/trials.csv --output results/plots

When running receiver.py, set Trial note as one of:
    experiment=bit_duration; trial=1
    experiment=tone_separation; trial=1
    experiment=background_interference; setting=quiet; trial=1
    experiment=timing_offset; trial=1

The script reads receiver.py's CSV without changing it. It does NOT
simulate experiments, generate BER values, or impose a trial limit.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")  # Save PNGs without opening a graphical window.
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "results" / "trials.csv"
DEFAULT_OUTPUT = ROOT / "results" / "plots"

EXPERIMENTS = {
    "bit_duration": ("Bit Duration (s)", "numeric"),
    "tone_separation": ("Tone Separation (Hz)", "numeric"),
    "background_interference": ("Background Interference", "category"),
    "timing_offset": ("Timing Offset (ms)", "numeric"),
}


def parse_note(note):
    """Return key=value labels separated by semicolons in Trial note."""
    return {
        key.strip().lower(): value.strip()
        for key, value in re.findall(r"(?:^|;)\s*([\w-]+)\s*=\s*([^;]+)", note or "")
    }


def get_ber(row):
    """Prefer exact errors/n_bits; receiver.py rounds its stored ber."""
    if row.get("errors") not in (None, "") and row.get("n_bits") not in (None, ""):
        errors = float(row["errors"])
        count = float(row["n_bits"])
        if count <= 0 or errors < 0 or errors > count:
            raise ValueError("Invalid errors or n_bits")
        return errors / count

    if row.get("ber") in (None, ""):
        raise ValueError("Missing errors/n_bits and ber")
    value = float(row["ber"])
    if not 0 <= value <= 1:
        raise ValueError("BER is outside 0..1")
    return value
def get_setting(row, experiment, labels):
    """Read measured settings from actual CSV columns, not from guessed notes."""
    if experiment == "bit_duration":
        return float(row["bit_duration_s"])
    if experiment == "tone_separation":
        return abs(float(row["f1_hz"]) - float(row["f0_hz"]))
    if experiment == "timing_offset":
        return float(row["offset_ms"])
    if experiment == "background_interference":
        setting = labels.get("setting", "").strip().lower()
        if not setting:
            raise ValueError("Background interference needs setting=quiet/talking/music in Trial note")
        return setting
    raise ValueError("Unsupported experiment")


def load_groups(csv_path):
    groups = {name: defaultdict(list) for name in EXPERIMENTS}
    ignored = []

    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"note", "ber", "bit_duration_s", "f0_hz", "f1_hz", "offset_ms"}
        fields = set(reader.fieldnames or ())
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(missing)}")

        for line_number, row in enumerate(reader, start=2):
            labels = parse_note(row.get("note", ""))
            experiment = labels.get("experiment", "").lower()
            if experiment not in EXPERIMENTS:
                ignored.append((line_number, "Missing/unknown experiment label in Trial note"))
                continue
            try:
                setting = get_setting(row, experiment, labels)
                ber = get_ber(row)
            except (ValueError, KeyError, TypeError) as exc:
                ignored.append((line_number, str(exc)))
                continue
            groups[experiment][setting].append(ber)

    return groups, ignored


def make_plot(experiment, settings, output_dir):
    xlabel, style = EXPERIMENTS[experiment]
    if not settings:
        print(f"SKIP {experiment}: no labelled valid trials")
        return

    if style == "numeric":
        keys = sorted(settings)
    else:
        preferred = ["quiet", "talking", "music"]
        keys = [k for k in preferred if k in settings]
        keys += sorted(k for k in settings if k not in keys)

    averages = [mean(settings[key]) for key in keys]
    variability = [stdev(settings[key]) if len(settings[key]) > 1 else 0.0 for key in keys]
    counts = [len(settings[key]) for key in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    if style == "numeric":
        ax.errorbar(keys, averages, yerr=variability, fmt="o-", capsize=5, linewidth=2)
        ax.set_xticks(keys)
    else:
        x = list(range(len(keys)))
        ax.bar(x, averages, yerr=variability, capsize=5)
        ax.set_xticks(x, labels=[str(k).title() for k in keys])

    # Counts disclose how many real trials support each point/bar.
    for x_value, avg, count in zip((keys if style == "numeric" else range(len(keys))), averages, counts):
        ax.annotate(f"n={count}", (x_value, avg), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)

    ax.set_title(experiment.replace("_", " ").title() + " vs BER")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean Bit Error Rate (BER)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output = output_dir / f"{experiment}_ber.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print(f"SAVED {output}")
    for key, avg, count in zip(keys, averages, counts):
        print(f"  {str(key):>12}  trials={count:<4} mean BER={avg:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot Acoustic Messenger experiment BER")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="CSV created by receiver.py (default: results/trials.csv)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output folder for four graphs (default: results/plots)")
    args = parser.parse_args()

    if not args.csv.is_file():
        parser.error(f"File not found: {args.csv}. Run receiver.py first.")

    try:
        groups, ignored = load_groups(args.csv)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
        
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Reading: {args.csv}")
    for experiment, settings in groups.items():
        make_plot(experiment, settings, args.output)

    if ignored:
        print(f"\nIgnored {len(ignored)} row(s):")
        for number, reason in ignored[:15]:
            print(f"  CSV line {number}: {reason}")
        if len(ignored) > 15:
            print("  ...")
        print("Use Trial note: experiment=bit_duration; trial=1")
        print("For noise: experiment=background_interference; setting=quiet; trial=1")


if __name__ == "__main__":
    main()
