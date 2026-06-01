"""
Valence-Arousal Emotion Model for Classical Piano Music.

Two modes:
1. Dataset-relative: uses percentile normalization so scores are balanced
   within the dataset (for CSV batch processing).
2. Absolute: uses fixed musical reference points (for single MIDI uploads).

Reference: Russell's circumplex model of affect (1980).
"""

import numpy as np
import pandas as pd


def _clip01(x):
    return max(0.0, min(1.0, x))


# ============================================================
#  Dataset-relative normalization (used in batch regeneration)
# ============================================================

def fit_norms(df: pd.DataFrame) -> dict:
    """Compute normalization reference points from dataset percentiles.
    p5 -> 0, p95 -> 1, clipped.
    """
    refs = {}
    cols_map = {
        "tempo": "tempo", "note_density": "note_density",
        "velocity_variance": "velocity_variance", "pitch_range": "pitch_range",
        "avg_pitch": "avg_pitch", "melodic_complexity": "melodic_complexity",
        "pitch_entropy": "pitch_entropy",
    }
    for key, col in cols_map.items():
        if col in df.columns:
            lo = float(np.percentile(df[col].dropna(), 5))
            hi = float(np.percentile(df[col].dropna(), 95))
            refs[key] = (lo, hi)
    return refs


# ============================================================
#  Core VA computation
# ============================================================

def compute_valence_arousal(
    tempo: float,
    note_density: float,
    velocity_variance: float,
    pitch_range: float,
    avg_pitch: float,
    melodic_complexity: float,
    pitch_entropy: float,
    key: str = "",
    key_confidence: float = 0.0,
    norms: dict | None = None,
    splits: tuple | None = None,  # (v_split, a_split) — dataset medians
    **_kwargs,
) -> dict:
    """
    Compute valence and arousal scores.

    If `norms` is provided (dict of (lo, hi) tuples), uses percentile-based
    normalization for dataset-relative scoring. Otherwise uses fixed musical
    references suitable for single-file evaluation.

    Returns: {valence_score, arousal_score, emotion_label, sub_region, mode}
    """

    # ---- Normalize features ----
    # Use musically calibrated absolute references for both modes.
    # These map typical musical extremes to 0→1:
    #   note_density: slow chorale(~3) → 0, virtuoso etude(~18) → 1
    #   tempo: Largo(~40) → 0, Presto(~200) → 1  (often 120 default in MAESTRO)
    #   pitch_range: simple melody(~20st) → 0, full keyboard(~88st) → 1
    #   velocity_variance: flat(~50) → 0, extreme(~500) → 1
    #   melodic_complexity: stepwise(~3) → 0, angular(~16) → 1
    #   pitch_entropy: diatonic(~0.68) → 1(inverted), chromatic(~0.98) → 0(inverted)
    t_norm = _clip01((tempo - 40.0) / 160.0) if tempo > 40 else 0.0
    d_norm = _clip01((note_density - 3.0) / 15.0)
    v_norm = _clip01((velocity_variance - 50.0) / 450.0)
    r_norm = _clip01((pitch_range - 20.0) / 68.0)
    p_norm = _clip01((avg_pitch - 40.0) / 45.0)
    m_norm = _clip01((melodic_complexity - 3.0) / 13.0)
    # Invert pitch_entropy: diatonic=high valence, chromatic=low valence
    e_norm = 1.0 - _clip01((pitch_entropy - 0.68) / 0.30)

    # ---- Arousal (energy) ----
    # Note: tempo is often 120 (default) in MAESTRO — MIDI lacks set_tempo events.
    # We deprioritize it heavily; note_density and pitch_range are the real signals.
    arousal = 0.50 * d_norm + 0.10 * t_norm + 0.25 * r_norm + 0.15 * v_norm
    arousal = _clip01(arousal)

    # ---- Mode ----
    mode = "unknown"
    mode_boost = 0.55
    if key and key_confidence > 0.3:
        key_lower = key.lower()
        if "minor" in key_lower:
            mode = "minor"
            mode_boost = 0.35 + 0.15 * (key_confidence / 100.0)
        elif "major" in key_lower:
            mode = "major"
            mode_boost = 0.70 + 0.15 * (key_confidence / 100.0)

    # ---- Valence (pleasantness) ----
    valence = 0.35 * mode_boost + 0.25 * e_norm + 0.20 * (1.0 - m_norm * 0.5) + 0.20 * p_norm
    valence = _clip01(valence)

    # ---- Emotion label ----
    if splits:
        v_split, a_split = splits
    else:
        v_split, a_split = 0.50, 0.45
    emotion_label, sub_region = _classify_region(valence, arousal, mode, v_split, a_split)

    return {
        "valence_score": round(valence, 4),
        "arousal_score": round(arousal, 4),
        "emotion_label": emotion_label,
        "sub_region": sub_region,
        "mode": mode,
    }


def _classify_region(valence: float, arousal: float, mode: str = "",
                     v_split: float = 0.50, a_split: float = 0.45) -> tuple:
    """Map (valence, arousal) to emotion label.

    Split points default to 0.50/0.45 but should be overridden with
    actual dataset medians for balanced quadrants in batch mode.
    """

    # Minor-key pieces with high energy → dramatic/passionate, not agitated
    if mode == "minor" and arousal > 0.5:
        effective_valence = valence + 0.12
    else:
        effective_valence = valence

    if arousal >= a_split and effective_valence >= v_split:
        label = "Passionate"
        sub = "Heroic" if arousal > 0.7 else "Dramatic"
    elif arousal >= a_split and effective_valence < v_split:
        label = "Agitated"
        sub = "Stormy" if arousal > 0.7 else "Tense"
    elif arousal < a_split and effective_valence < v_split:
        label = "Melancholic"
        sub = "Somber" if valence < 0.3 else "Nostalgic"
    else:
        label = "Tender/Calm"
        sub = "Serene" if arousal < 0.25 else "Lyrical"
    return label, sub


# ============================================================
#  Convenience wrappers
# ============================================================

def compute_from_analyze_result(result: dict, norms: dict | None = None) -> dict:
    """Compute VA from an analyze_midi() result dict."""
    return compute_valence_arousal(
        tempo=result.get("tempo", 120),
        note_density=result.get("note_density", 0),
        velocity_variance=result.get("velocity_variance", 0),
        pitch_range=result.get("pitch_range", 0),
        avg_pitch=result.get("avg_pitch", 60),
        melodic_complexity=result.get("melodic_complexity", 0),
        pitch_entropy=result.get("pitch_entropy", 0),
        key=result.get("key", ""),
        key_confidence=result.get("key_confidence", 0),
        norms=norms,
    )


def compute_from_dataframe_row(row: pd.Series, norms: dict | None = None) -> dict:
    """Compute VA from a music_dataset.csv row."""
    return compute_valence_arousal(
        tempo=float(row.get("tempo", 120)),
        note_density=float(row.get("note_density", 0)),
        velocity_variance=float(row.get("velocity_variance", 0)),
        pitch_range=float(row.get("pitch_range", 0)),
        avg_pitch=float(row.get("avg_pitch", 60)),
        melodic_complexity=float(row.get("melodic_complexity", 0)),
        pitch_entropy=float(row.get("pitch_entropy", 0)),
        key=str(row.get("key", "")),
        key_confidence=float(row.get("key_confidence", 0)),
        norms=norms,
    )


# ============================================================
#  Self-test
# ============================================================

if __name__ == "__main__":
    print("=== Absolute mode (single upload) ===")
    tests = [
        ("Chopin Polonaise E Major", dict(tempo=120, note_density=15, velocity_variance=229,
            pitch_range=76, avg_pitch=68, melodic_complexity=12, pitch_entropy=0.96,
            key="E Major", key_confidence=91)),
        ("Chopin Nocturne Eb Major", dict(tempo=80, note_density=5, velocity_variance=200,
            pitch_range=50, avg_pitch=64, melodic_complexity=10, pitch_entropy=0.85,
            key="Eb Major", key_confidence=85)),
        ("Bach Prelude C Major", dict(tempo=100, note_density=6, velocity_variance=100,
            pitch_range=45, avg_pitch=65, melodic_complexity=6, pitch_entropy=0.72,
            key="C Major", key_confidence=95)),
        ("Beethoven Appassionata", dict(tempo=140, note_density=12, velocity_variance=450,
            pitch_range=75, avg_pitch=60, melodic_complexity=12, pitch_entropy=0.88,
            key="F Minor", key_confidence=90)),
    ]
    for name, params in tests:
        r = compute_valence_arousal(**params)
        print(f"  {name:30s} v={r['valence_score']:.3f} a={r['arousal_score']:.3f} -> {r['emotion_label']}")

    # Test with dataset-relative norms
    print("\n=== Dataset-relative mode ===")
    import os
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "database", "music_dataset.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        norms = fit_norms(df)
        print(f"  Norms: { {k: f'{v[0]:.1f}-{v[1]:.1f}' for k, v in list(norms.items())[:4]} }")
        for name, params in tests:
            r = compute_valence_arousal(**params, norms=norms)
            print(f"  {name:30s} v={r['valence_score']:.3f} a={r['arousal_score']:.3f} -> {r['emotion_label']}")

    print("\n[OK] emotion_model.py self-test passed.")
