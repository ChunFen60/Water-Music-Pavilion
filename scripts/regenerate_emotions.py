"""
Regenerate the music dataset with Valence-Arousal emotion scores.

Reads each MIDI file referenced in music_dataset.csv, runs the full
analyze_midi() pipeline, computes VA emotion scores, and writes an
updated CSV with new columns: valence_score, arousal_score, emotion_label.

Usage:
    python scripts/regenerate_emotions.py
    python scripts/regenerate_emotions.py --workers 4
    python scripts/regenerate_emotions.py --dry-run   (print what would change)

The old 'emotion' column is preserved as 'emotion_legacy'.
"""

import os
import sys
import argparse
import time

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from modules.midi_analyzer import analyze_midi
from modules.emotion_model import compute_from_analyze_result, fit_norms, _classify_region

CSV_PATH = os.path.join(BASE_DIR, "database", "music_dataset.csv")
BACKUP_PATH = CSV_PATH.replace(".csv", "_backup_before_va.csv")
MAESTRO_BASE = os.path.join(BASE_DIR, "data", "maestro", "maestro-v3.0.0")


def resolve_midi_path(rel_path: str) -> str:
    """Convert CSV-relative MIDI path to absolute filesystem path."""
    abs_path = os.path.join(MAESTRO_BASE, rel_path)
    if os.path.exists(abs_path):
        return abs_path
    # Try normalizing path separators
    norm = os.path.join(MAESTRO_BASE, rel_path.replace("/", os.sep))
    if os.path.exists(norm):
        return norm
    return abs_path  # return best guess even if missing


def regenerate(csv_path: str = CSV_PATH, dry_run: bool = False) -> pd.DataFrame:
    """Main regeneration logic. Returns the updated DataFrame."""

    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    n_total = len(df)
    print(f"  {n_total} rows loaded.")

    # ---- fit dataset-relative norms ----
    norms = fit_norms(df)
    print(f"  VA norms fitted from existing columns.")
    # Add pitch_entropy norms manually (not in old CSV; use typical range)
    norms.setdefault("pitch_entropy", (0.70, 0.95))
    norms.setdefault("avg_pitch", (50.0, 78.0))

    # ---- backup ----
    if not dry_run:
        print(f"Backing up to: {BACKUP_PATH}")
        df.to_csv(BACKUP_PATH, index=False)

    # ---- new columns ----
    df["emotion_legacy"] = df.get("emotion", pd.Series([""] * n_total))

    valences = []
    arousals = []
    labels = []
    sub_regions = []
    modes = []
    # harmony features
    h_diversity = []
    h_rhythm = []
    h_td_ratio = []
    h_bass_step = []
    h_chromatic = []
    errors = 0
    skipped = 0

    t0 = time.time()

    for idx, row in df.iterrows():
        midi_rel = row.get("midi_path", "")
        midi_path = resolve_midi_path(str(midi_rel))
        title = str(row.get("title", "?"))[:60]

        if not os.path.exists(midi_path):
            print(f"  [{idx+1}/{n_total}] MISSING: {midi_rel}")
            valences.append(np.nan)
            arousals.append(np.nan)
            labels.append("Unknown")
            sub_regions.append("")
            modes.append("")
            h_diversity.append(0.0)
            h_rhythm.append(0.0)
            h_td_ratio.append(0.5)
            h_bass_step.append(0.0)
            h_chromatic.append(0.0)
            skipped += 1
            continue

        try:
            result = analyze_midi(midi_path, skip_harmony=True)
            if "error" in result:
                print(f"  [{idx+1}/{n_total}] ERROR: {title} -> {result['error']}")
                valences.append(np.nan)
                arousals.append(np.nan)
                labels.append("Unknown")
                sub_regions.append("")
                modes.append("")
                h_diversity.append(0.0)
                h_rhythm.append(0.0)
                h_td_ratio.append(0.5)
                h_bass_step.append(0.0)
                h_chromatic.append(0.0)
                errors += 1
                continue

            va = compute_from_analyze_result(result, norms=norms)
            valences.append(va["valence_score"])
            arousals.append(va["arousal_score"])
            labels.append("")  # placeholder, computed after medians
            sub_regions.append(va["sub_region"])  # placeholder
            modes.append(va["mode"])
            h_diversity.append(result.get("chord_diversity", 0.0))
            h_rhythm.append(result.get("harmonic_rhythm", 0.0))
            h_td_ratio.append(result.get("tonic_dominant_ratio", 0.5))
            h_bass_step.append(result.get("bass_motion_step_ratio", 0.0))
            h_chromatic.append(result.get("chromatic_pct", 0.0))

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                remaining = (n_total - idx - 1) / rate
                print(f"  [{idx+1}/{n_total}] {rate:.1f} files/s, ETA {remaining:.0f}s  "
                      f"last: {title} -> {va['emotion_label']}")

        except Exception as e:
            print(f"  [{idx+1}/{n_total}] EXCEPTION: {title} -> {e}")
            valences.append(np.nan)
            arousals.append(np.nan)
            labels.append("Unknown")
            sub_regions.append("")
            modes.append("")
            h_diversity.append(0.0)
            h_rhythm.append(0.0)
            h_td_ratio.append(0.5)
            h_bass_step.append(0.0)
            h_chromatic.append(0.0)
            errors += 1

    # ---- post-process: recompute labels using data medians for balanced quadrants ----
    valid_mask = ~np.isnan(valences)
    v_vals = np.array(valences)[valid_mask]
    a_vals = np.array(arousals)[valid_mask]
    if len(v_vals) > 0:
        v_med = float(np.median(v_vals))
        a_med = float(np.median(a_vals))
    else:
        v_med, a_med = 0.50, 0.45
    print(f"  Median valence={v_med:.3f}, Median arousal={a_med:.3f}")

    for i in range(len(labels)):
        if not np.isnan(valences[i]):
            label, sub = _classify_region(valences[i], arousals[i], modes[i], v_med, a_med)
            labels[i] = label
            sub_regions[i] = sub
        else:
            labels[i] = "Unknown"
            sub_regions[i] = ""

    # ---- assign columns ----
    df["valence_score"] = valences
    df["arousal_score"] = arousals
    df["emotion_label"] = labels
    df["emotion_sub_region"] = sub_regions
    df["detected_mode"] = modes
    df["chord_diversity"] = h_diversity
    df["harmonic_rhythm"] = h_rhythm
    df["tonic_dominant_ratio"] = h_td_ratio
    df["bass_motion_step_ratio"] = h_bass_step
    df["chromatic_pct"] = h_chromatic

    # drop old emotion column (we already have emotion_legacy)
    if "emotion" in df.columns:
        df = df.drop(columns=["emotion"])

    # ---- save ----
    if not dry_run:
        print(f"\nSaving updated dataset to: {csv_path}")
        df.to_csv(csv_path, index=False)
        print("Done.")

    # ---- report ----
    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"Regeneration complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Total: {n_total}  |  Errors: {errors}  |  Skipped (missing): {skipped}")
    print(f"\nEmotion distribution (new labels):")
    for label, count in df["emotion_label"].value_counts().items():
        pct = 100 * count / n_total
        print(f"  {label:20s}: {count:5d}  ({pct:5.1f}%)")
    print(f"\nVA score ranges:")
    print(f"  Valence: [{df['valence_score'].min():.3f}, {df['valence_score'].max():.3f}]  "
          f"mean={df['valence_score'].mean():.3f}")
    print(f"  Arousal: [{df['arousal_score'].min():.3f}, {df['arousal_score'].max():.3f}]  "
          f"mean={df['arousal_score'].mean():.3f}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate music dataset with VA emotion scores"
    )
    parser.add_argument("--csv", default=CSV_PATH, help="Path to music_dataset.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without saving")
    parser.add_argument("--workers", type=int, default=1,
                        help="(Reserved for future multiprocessing)")
    args = parser.parse_args()

    regenerate(csv_path=args.csv, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
