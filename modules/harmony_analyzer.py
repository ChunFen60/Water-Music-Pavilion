"""
Harmony Analysis Module for Classical Piano MIDI.

Extracts harmonic features from note-level MIDI data:
- Chord detection via template matching (10 chord types × 12 roots)
- Roman numeral analysis (key-aware)
- Cadence detection (PAC, IAC, HC, DC, Plagal, Deceptive)
- Harmonic rhythm and chord diversity metrics
- Bass motion analysis

All algorithms are pure numpy — no external music theory libraries needed.
"""

import numpy as np

# ============================================================
#  Chord Templates  (12-element pitch-class vectors)
# ============================================================
# Each template represents the pitch-class content of a chord type,
# starting from root=0 (C).  Rotated for all 12 roots during matching.

CHORD_TEMPLATES = {
    "M":     np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float),   # Major
    "m":     np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float),   # Minor
    "dim":   np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=float),   # Diminished
    "aug":   np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=float),   # Augmented
    "dom7":  np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=float),   # Dominant 7th
    "maj7":  np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=float),   # Major 7th
    "min7":  np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=float),   # Minor 7th
    "dim7":  np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=float),   # Diminished 7th
    "hdim7": np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=float),   # Half-dim 7th
    "sus4":  np.array([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], dtype=float),   # Suspended 4th
}

# Friendly labels for display
CHORD_DISPLAY = {
    "M": "Major", "m": "Minor", "dim": "Dim", "aug": "Aug",
    "dom7": "Dom7", "maj7": "Maj7", "min7": "Min7",
    "dim7": "Dim7", "hdim7": "Half-dim7", "sus4": "Sus4",
}

# Scale degree names for Roman numerals
SCALE_DEGREE_MAJOR = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
SCALE_DEGREE_MINOR = ["i", "ii°", "III", "iv", "V", "VI", "vii°"]

# Quality-based Roman numeral adjustments
ROMAN_QUALITY = {
    "M": "", "m": "", "dim": "°", "aug": "+",
    "dom7": "⁷", "maj7": "Δ⁷", "min7": "⁷",
    "dim7": "°⁷", "hdim7": "ø⁷", "sus4": "sus⁴",
}

# Interval (in semitones) from each scale degree to tonic
MAJOR_INTERVALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
MINOR_INTERVALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Scale degree → semitones from tonic (Major)
DEGREE_TO_SEMITONE_MAJOR = {0: 0, 1: 2, 2: 4, 3: 5, 4: 7, 5: 9, 6: 11}
# Scale degree → semitones from tonic (Minor, natural)
DEGREE_TO_SEMITONE_MINOR = {0: 0, 1: 2, 2: 3, 3: 5, 4: 7, 5: 8, 6: 10}

# Cadence patterns: (chord_sequence, label)
# Each entry is a list of (degree_offset, quality) tuples
CADENCE_PATTERNS = [
    ([(5, "M"), (0, "M")], "PAC/IAC"),       # V → I
    ([(5, "dom7"), (0, "M")], "PAC/IAC"),     # V⁷ → I
    ([(5, "M"), (0, "m")], "PAC/IAC"),        # V → i
    ([(5, "dom7"), (0, "m")], "PAC/IAC"),     # V⁷ → i
    ([(5, "M")], "HC"),                       # ending on V (Half Cadence)
    ([(5, "dom7")], "HC"),                    # ending on V⁷
    ([(4, "M"), (0, "M")], "Plagal"),         # IV → I
    ([(4, "m"), (0, "m")], "Plagal"),         # iv → i
    ([(4, "M"), (0, "m")], "Plagal"),         # IV → i
    ([(5, "M"), (6, "m")], "Deceptive"),      # V → vi
    ([(5, "dom7"), (6, "m")], "Deceptive"),   # V⁷ → vi
    ([(5, "M"), (6, "M")], "Deceptive"),      # V → VI
    ([(5, "dom7"), (6, "M")], "Deceptive"),   # V⁷ → VI
]


# ============================================================
#  Pitch-class extraction
# ============================================================

def _pitch_classes_in_window(notes_data: list, t_start: float, t_end: float) -> np.ndarray:
    """Return 12-element duration-weighted pitch-class vector for a time window."""
    vec = np.zeros(12, dtype=float)
    for n in notes_data:
        if n["start"] < t_end and n["end"] > t_start:
            overlap = min(n["end"], t_end) - max(n["start"], t_start)
            if overlap > 0:
                weight = overlap * (n.get("velocity", 64) / 127.0)
                vec[n["pitch"] % 12] += weight
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


# ============================================================
#  Chord matching
# ============================================================

def _match_chord(pc_vector: np.ndarray, min_confidence: float = 0.35) -> dict | None:
    """Match a pitch-class vector to the closest chord template.

    Returns: {"root": int, "type": str, "confidence": float, "roman": str} or None
    """
    best_sim = -1.0
    best_type = None
    best_root = 0

    for ctype, template in CHORD_TEMPLATES.items():
        for root in range(12):
            rotated = np.roll(template, root)
            # Cosine similarity
            dot = np.dot(pc_vector, rotated)
            norms = np.linalg.norm(pc_vector) * np.linalg.norm(rotated)
            if norms < 1e-10:
                continue
            sim = dot / norms
            if sim > best_sim:
                best_sim = sim
                best_type = ctype
                best_root = root

    if best_sim < min_confidence or best_type is None:
        return None

    return {
        "root": best_root,
        "type": best_type,
        "confidence": round(float(best_sim), 3),
        "roman": "",  # filled in later
    }


# ============================================================
#  Roman numeral mapping
# ============================================================

def _key_to_tonic(key: str) -> int:
    """Extract tonic pitch class (0=C) from key name string."""
    key_map = {
        "c": 0, "c#": 1, "db": 1, "d": 2, "d#": 3, "eb": 3,
        "e": 4, "f": 5, "f#": 6, "gb": 6, "g": 7, "g#": 8, "ab": 8,
        "a": 9, "a#": 10, "bb": 10, "b": 11,
    }
    key_lower = key.strip().lower().split()[0] if key else "c"
    return key_map.get(key_lower, 0)


def _is_major(key: str) -> bool:
    return "major" in key.lower()


def _roman_numeral(root_pc: int, chord_type: str, key: str) -> str:
    """Convert (root pitch class, chord type, key) to Roman numeral."""
    tonic = _key_to_tonic(key)
    major = _is_major(key)

    # Semitones from tonic
    interval = (root_pc - tonic) % 12

    # Map interval → scale degree
    if major:
        dmap = {v: k for k, v in DEGREE_TO_SEMITONE_MAJOR.items()}
        degree_names = SCALE_DEGREE_MAJOR
    else:
        dmap = {v: k for k, v in DEGREE_TO_SEMITONE_MINOR.items()}
        degree_names = SCALE_DEGREE_MINOR

    if interval in dmap:
        degree = dmap[interval]
        rn = degree_names[degree]
    else:
        # Chromatic / non-diatonic — show as bII, #IV, etc.
        closest = min(dmap.keys(), key=lambda x: abs(x - interval))
        degree = dmap[closest]
        diff = interval - closest
        if diff > 0:
            prefix = "#" if diff == 1 else f"+{diff}"
            rn = f"{prefix}{degree_names[degree]}"
        elif diff < 0:
            prefix = "b" if diff == -1 else f"{diff}"
            rn = f"{prefix}{degree_names[degree]}"
        else:
            rn = degree_names[degree]

    qual = ROMAN_QUALITY.get(chord_type, "")
    return f"{rn}{qual}"


# ============================================================
#  Cadence detection
# ============================================================

def _detect_cadences(
    chord_sequence: list,
    notes_data: list,
    gap_threshold: float = 0.5,
) -> list:
    """Find cadences by scanning chord sequence near phrase boundaries.

    A phrase boundary is a point where no notes are active (silence gap).
    We look at the last 2-3 chords before each silence for cadence patterns.
    """
    if len(chord_sequence) < 2:
        return []

    # Find silence gaps (phrase boundaries)
    events = sorted(
        [(n["start"], 1) for n in notes_data] +
        [(n["end"], -1) for n in notes_data],
        key=lambda x: x[0]
    )
    active = 0
    gaps = []  # (start_time, end_time)
    gap_start = None
    for t, delta in events:
        was_active = active > 0
        active += delta
        is_active = active > 0
        if was_active and not is_active:
            gap_start = t
        elif not was_active and is_active:
            if gap_start is not None and (t - gap_start) > gap_threshold:
                gaps.append((gap_start, t))
            gap_start = None

    cadences = []
    for gap_start, gap_end in gaps:
        # Find the last few chords before this gap
        chords_before = [c for c in chord_sequence if c["start"] < gap_start]
        if len(chords_before) < 2:
            continue

        # Take last 2 chords
        last_two = chords_before[-2:]

        # Check against cadence patterns
        for pattern, label in CADENCE_PATTERNS:
            if len(last_two) < len(pattern):
                continue
            recent = last_two[-len(pattern):]
            match = True
            for (exp_root, exp_type), actual in zip(pattern, recent):
                actual_root = actual["root"]
                actual_type = actual["type"]
                # Compare scale degree (not absolute root) against pattern
                actual_degree = actual.get("scale_degree", actual_root)
                if actual_degree != exp_root or actual_type != exp_type:
                    match = False
                    break
            if match:
                cadences.append({
                    "time": round(gap_start, 2),
                    "type": label,
                    "strength": round(
                        sum(c["confidence"] for c in recent) / len(recent), 3
                    ),
                })
                break  # one cadence per gap

    return cadences


# ============================================================
#  Main analysis function
# ============================================================

def _match_chord_fast(pc_vector: np.ndarray, template_rots: dict, min_confidence: float = 0.35) -> dict | None:
    """Match pitch-class vector using pre-computed template rotations."""
    best_sim = -1.0
    best_type = None
    best_root = 0
    norm_pc = np.linalg.norm(pc_vector)
    if norm_pc < 1e-10:
        return None

    for ctype, rots in template_rots.items():
        for root, rotated in rots:
            dot = np.dot(pc_vector, rotated)
            norm_t = np.linalg.norm(rotated)
            if norm_t < 1e-10:
                continue
            sim = dot / (norm_pc * norm_t)
            if sim > best_sim:
                best_sim = sim
                best_type = ctype
                best_root = root

    if best_sim < min_confidence or best_type is None:
        return None

    return {"root": best_root, "type": best_type, "confidence": round(float(best_sim), 3), "roman": ""}


def analyze_harmony(notes_data: list, detected_key: str = "") -> dict:
    """Full harmonic analysis of note-level MIDI data.

    Args:
        notes_data: list of {pitch, start, end, duration, velocity} dicts
        detected_key: key string from key_detection (e.g. "C Major")

    Returns:
        dict with chord_types, chord_diversity, harmonic_rhythm,
        cadence_counts, bass_motion_step_ratio, tonic_dominant_ratio
    """
    if not notes_data:
        return _empty_result()

    # ---- estimate window size from median IOI ----
    starts = sorted(n["start"] for n in notes_data)
    iois = np.diff(starts)
    iois = iois[iois > 0]
    median_ioi = float(np.median(iois)) if len(iois) > 0 else 0.1

    # Window = 8th note equivalent (2x median IOI, clamped)
    window = max(0.08, min(0.5, median_ioi * 2.0))

    # ---- compute polyphony to check if polyphonic ----
    events = []
    for n in notes_data:
        events.append((n["start"], 1))
        events.append((n["end"], -1))
    events.sort(key=lambda x: x[0])
    cur = 0
    weighted = 0.0
    prev = events[0][0] if events else 0
    for t, d in events:
        weighted += cur * (t - prev)
        prev = t
        cur += d
    total_t = events[-1][0] - events[0][0] if events else 1
    avg_polyphony = weighted / total_t if total_t > 0 else 0

    # Monophonic fallback — no harmonic analysis possible
    if avg_polyphony < 1.3:
        return _empty_result(reason="monophonic")

    # ---- Slice into windows and detect chords (event-sweep for speed) ----
    t_min = notes_data[0]["start"]
    t_max = max(n["end"] for n in notes_data)
    total_duration = t_max - t_min or 1.0

    # Build sorted timeline of note start/end events for fast window extraction
    note_events = []  # (time, delta, pitch, velocity)
    for n in notes_data:
        note_events.append((n["start"], +1, n["pitch"], n.get("velocity", 64)))
        note_events.append((n["end"],   -1, n["pitch"], n.get("velocity", 64)))
    note_events.sort(key=lambda x: x[0])

    # Pre-allocate chord templates at all rotations (compute once)
    template_rots = {}
    for ctype, tmpl in CHORD_TEMPLATES.items():
        rots = []
        for r in range(12):
            rots.append((r, np.roll(tmpl, r)))
        template_rots[ctype] = rots

    chord_sequence = []
    t = t_min
    prev_chord = None
    ev_idx = 0  # pointer into note_events
    active_pc = np.zeros(12, dtype=float)  # running pitch-class accumulator
    active_weight = np.zeros(12, dtype=float)

    while t < t_max:
        t_end = t + window

        # Fast-forward: remove notes that ended before this window
        while ev_idx < len(note_events) and note_events[ev_idx][0] < t:
            _, delta, pitch, vel = note_events[ev_idx]
            pc = pitch % 12
            w = 1.0  # unweighted count for speed
            if delta < 0:
                active_pc[pc] -= w
                active_weight[pc] -= w
            else:
                active_pc[pc] += 1  # will be replaced next sweep
            ev_idx += 1

        # Rewind to catch notes active during [t, t_end)
        # Use temp pointer to scan forward from ev_idx
        temp_idx = ev_idx
        temp_pc = active_pc.copy()
        while temp_idx < len(note_events) and note_events[temp_idx][0] < t_end:
            _, delta, pitch, vel = note_events[temp_idx]
            pc = pitch % 12
            if delta > 0:  # note starts within window
                temp_pc[pc] += 1
            temp_idx += 1

        # Normalize and match
        total = temp_pc.sum()
        if total > 0:
            pc_vec = temp_pc / total
        else:
            pc_vec = temp_pc

        chord = _match_chord_fast(pc_vec, template_rots)

        if chord is not None:
            chord["start"] = round(t, 2)
            chord["end"] = round(t_end, 2)
            if detected_key:
                chord["roman"] = _roman_numeral(chord["root"], chord["type"], detected_key)

            if prev_chord and prev_chord["root"] == chord["root"] and prev_chord["type"] == chord["type"]:
                prev_chord["end"] = chord["end"]
            else:
                chord_sequence.append(chord)
                prev_chord = chord

        t += window

    # ---- Chord type distribution ----
    chord_types = {display: 0 for display in CHORD_DISPLAY.values()}
    chord_types["Unknown"] = 0
    total_chords = len(chord_sequence)
    for c in chord_sequence:
        display_name = CHORD_DISPLAY.get(c["type"], c["type"])
        chord_types[display_name] += 1

    # Remove zero-count types
    chord_types = {k: v for k, v in chord_types.items() if v > 0}

    # ---- Chord diversity ----
    unique_types = len(chord_types)
    chord_diversity = unique_types / max(total_chords, 1)

    # ---- Harmonic rhythm (chord changes per second) ----
    # Count actual changes (ignore consecutive same)
    changes = sum(
        1 for i in range(1, len(chord_sequence))
        if chord_sequence[i]["root"] != chord_sequence[i-1]["root"] or
        chord_sequence[i]["type"] != chord_sequence[i-1]["type"]
    )
    harmonic_rhythm = changes / total_duration if total_duration > 0 else 0

    # ---- Cadence detection ----
    if detected_key:
        # Add Roman numerals and scale degrees first
        tonic = _key_to_tonic(detected_key)
        for c in chord_sequence:
            c["scale_degree"] = (c["root"] - tonic) % 12
        cadences = _detect_cadences(chord_sequence, notes_data)
    else:
        cadences = []

    cadence_counts = {"PAC/IAC": 0, "HC": 0, "DC": 0, "Plagal": 0, "Deceptive": 0}
    for cad in cadences:
        ctype = cad["type"]
        cadence_counts[ctype] = cadence_counts.get(ctype, 0) + 1

    # ---- Tonic / Dominant ratio ----
    tonic_count = sum(1 for c in chord_sequence if c["root"] == 0)
    dominant_count = sum(1 for c in chord_sequence if c["root"] == 5)
    total_with_root = max(tonic_count + dominant_count, 1)
    tonic_dominant_ratio = tonic_count / total_with_root

    # ---- Bass motion ----
    if len(chord_sequence) >= 2:
        steps = leaps = static = 0
        for i in range(1, len(chord_sequence)):
            interval = abs(chord_sequence[i]["root"] - chord_sequence[i-1]["root"])
            interval = min(interval, 12 - interval)  # shortest path
            if interval == 0:
                static += 1
            elif interval <= 2:
                steps += 1
            else:
                leaps += 1
        total_motions = steps + leaps + static
        bass_step_ratio = steps / max(total_motions, 1)
    else:
        bass_step_ratio = 0.0

    # ---- Percentage of unknown chords (atonal indicator) ----
    unknown_count = chord_types.get("Unknown", 0)
    chromatic_pct = unknown_count / max(total_chords, 1)

    return {
        "chord_sequence": chord_sequence[:50],  # cap for performance
        "chord_types": chord_types,
        "chord_diversity": round(chord_diversity, 4),
        "harmonic_rhythm": round(harmonic_rhythm, 4),
        "cadences": cadences,
        "cadence_counts": cadence_counts,
        "bass_motion_step_ratio": round(bass_step_ratio, 4),
        "tonic_dominant_ratio": round(tonic_dominant_ratio, 4),
        "chromatic_pct": round(chromatic_pct, 4),
        "total_chords_detected": total_chords,
    }


def _empty_result(reason: str = "") -> dict:
    return {
        "chord_sequence": [],
        "chord_types": {},
        "chord_diversity": 0.0,
        "harmonic_rhythm": 0.0,
        "cadences": [],
        "cadence_counts": {"PAC/IAC": 0, "HC": 0, "DC": 0, "Plagal": 0, "Deceptive": 0},
        "bass_motion_step_ratio": 0.0,
        "tonic_dominant_ratio": 0.5,
        "chromatic_pct": 0.0,
        "total_chords_detected": 0,
        "harmony_note": reason if reason else "",
    }


# ============================================================
#  Self-test
# ============================================================

if __name__ == "__main__":
    import sys, os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, BASE_DIR)

    from modules.midi_analyzer import analyze_midi

    # Find a test MIDI file
    test_path = os.path.join(BASE_DIR, "data", "maestro", "maestro-v3.0.0")
    for root, dirs, files in os.walk(test_path):
        for f in files:
            if f.endswith(".midi") or f.endswith(".mid"):
                midi_path = os.path.join(root, f)
                break
        else:
            continue
        break

    print(f"Testing with: {midi_path}")
    result = analyze_midi(midi_path)
    if "error" in result:
        print(f"ERROR: {result['error']}")
    else:
        notes = result["notes_data"]
        key = result["key"]
        print(f"Key: {key}, Notes: {len(notes)}")
        harmony = analyze_harmony(notes, key)
        print(f"\nChord types: {harmony['chord_types']}")
        print(f"Chord diversity: {harmony['chord_diversity']}")
        print(f"Harmonic rhythm: {harmony['harmonic_rhythm']:.3f} changes/sec")
        print(f"Cadence counts: {harmony['cadence_counts']}")
        print(f"Bass step ratio: {harmony['bass_motion_step_ratio']:.3f}")
        print(f"Tonic/Dominant ratio: {harmony['tonic_dominant_ratio']:.3f}")
        print(f"Chromatic pct: {harmony['chromatic_pct']:.3f}")
        print(f"\nFirst 5 chords:")
        for c in harmony["chord_sequence"][:5]:
            root_name = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"][c["root"]]
            print(f"  {root_name} {c['type']:6s} ({c['roman']:10s}) conf={c['confidence']:.2f}".encode('ascii', 'replace').decode())
        print("\n[OK] harmony_analyzer.py self-test passed.")
