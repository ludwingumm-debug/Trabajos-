#!/usr/bin/env python3
"""
BLACK ASPHALT - an original dark drift phonk track, synthesized from scratch.

Only numpy + scipy: every sound (cowbell, 808, kick, snare, hats, pads, lead,
risers, impacts, and the orchestral layer: strings, brass, choir, taiko,
timpani and the trailer "braam") is generated here; no samples are used.

    150 BPM · 4/4 · key of F# minor (with the dark phrygian G) · stereo · 44.1 kHz

Arrangement (60 bars, about 1:40):
    intro   8 bars  filtered cowbell, choir swell, solo strings hint the epic theme
    build   4 bars  snare + timpani rolls, tremolo strings, riser, one beat of silence
    drop 1 16 bars  braam, deep 808 dive, full beat, brass stabs, epic theme on strings
    break   8 bars  tape stop, choir, epic theme slowed down on strings, sub 808
    build   4 bars  second build
    drop 2 16 bars  climax: theme on strings + brass, choir, taiko, heavier beat
    outro   4 bars  cowbell fades, final braam + brass chord

Usage:
    python dark_drift_phonk.py              -> trap version  -> black_asphalt_trap.wav
    python dark_drift_phonk.py phonk        -> drift phonk + orchestra -> dark_drift_phonk.wav
    python dark_drift_phonk.py phonk himno  -> phonk version with another epic theme
                                               (original, himno, villano, heroica)
"""

import sys

import numpy as np
from scipy import ndimage, signal
from scipy.io import wavfile

# =============================================================================
# 1) GLOBAL SETTINGS
# =============================================================================
SR = 44100
BPM = 150                       # tempo (everything else follows it)
OUTPUT_FILE = "dark_drift_phonk.wav"
SEED = 1337                     # noise seed: same value = same track
TAIL = 4.0                      # seconds of reverb tail after the last bar

BEAT = 60.0 / BPM
STEP = BEAT / 4                 # one 16th note
BAR = BEAT * 4

# Key: every pitch is written in semitones relative to these F# roots.
ROOT_808 = 46.25                # F#1  - the 808 and kick fundamental
ROOT_PAD = ROOT_808 * 4         # F#3  - pad chords
ROOT_LEAD = ROOT_808 * 8        # F#4  - break lead
ROOT_COWBELL = ROOT_808 * 16    # F#5  - low tone of the cowbell (~740 Hz)
COWBELL_RATIO = 800 / 540       # ratio between the two cowbell oscillators

# Arrangement: (section name, bars). Reorder or resize freely.
SECTIONS = [
    ("intro", 8),
    ("build1", 4),
    ("drop1", 16),
    ("break", 8),
    ("build2", 4),
    ("drop2", 16),
    ("outro", 4),
]

# =============================================================================
# 2) MELODIES AND HARMONY  (semitones from the roots above; None = rest)
# =============================================================================
# Harmony: 4-bar cycle i - i - bII - bVII  (F#m, F#m, G, Em). The bII (G) is
# the phrygian half step that gives the track its dark, mysterious colour.
PROGRESSION = [0, 0, 1, -2]
PAD_CHORDS = [[0, 3, 7], [0, 3, 7], [1, 5, 8], [-2, 1, 5]]

# Cowbell riff A (drop 1, intro, builds): 4 bars of 16ths, call and answer.
_ = None
RIFF_A = [
    0, _, 0, 3,   _, 0, 1, _,   0, _, -2, _,   -5, _, -4, _,
    0, _, 0, 3,   _, 0, 1, _,   3, _, 5, 3,    1, _, 0, _,
    0, _, 0, 3,   _, 0, 1, _,   0, _, -2, _,   -5, _, -4, _,
    0, _, 0, 3,   _, 0, 1, _,   7, _, 8, 7,    5, 3, 1, _,
]
# Cowbell riff B (drop 2): an octave up, falling lines, more aggressive.
RIFF_B = [
    12, _, 12, 10,  _, 8, 7, _,   8, _, 7, _,    5, _, 3, _,
    1, _, 1, 3,     _, 1, 0, _,   -2, _, 0, _,   1, 3, 1, _,
    12, _, 12, 10,  _, 8, 7, _,   8, _, 7, _,    5, _, 3, _,
    12, _, 15, 13,  _, 12, 10, _, 8, 7, 8, _,    7, _, 1, 0,
]
# Break lead: (start in beats, semitones from F#4, length in beats). 8 bars.
LEAD_MELODY = [
    (0, 0, 2), (2, 3, 1.5), (3.5, 1, 0.5), (4, 0, 3), (7, -2, 1),
    (8, 1, 2), (10, 5, 2), (12, 3, 3), (15, 1, 1),
    (16, 0, 2), (18, 3, 1.5), (19.5, 5, 0.5), (20, 7, 3), (23, 8, 1),
    (24, 7, 2), (26, 5, 2), (28, 1, 3), (31, 0, 1),
]

# EPIC THEMES (orchestral): 8 bars over the 4-bar progression, played twice
# per drop. Each note: (start in beats, semitones from F#4, length in beats).
# Semitones: 0 F#4, 1 G, 3 A, 5 B, 7 C#5, 8 D, 10 E, 11 E#, 12 F#5, 13 G, 15 A,
#            17 B, 19 C#6, 20 D6.  Pick one with THEME (or the command line).
EPIC_THEMES = {}
# "original": climbs from F#4 to a peak on B5 in bar 7, falls back to E5.
EPIC_THEMES["original"] = [
    (0, 0, 1.5), (1.5, 7, 0.5), (2, 12, 2),            # F#m: F#4 - C#5 - F#5 (fifth + octave leap)
    (4, 10, 1), (5, 8, 1), (6, 7, 2),                  # F#m: E5 - D5 - C#5
    (8, 13, 1.5), (9.5, 12, 0.5), (10, 8, 2),          # G:   G5 - F#5 - D5
    (12, 10, 2), (14, 5, 1), (15, 7, 1),               # Em:  E5 - B4 - C#5
    (16, 12, 3), (19, 15, 1),                          # F#m: F#5 - A5
    (20, 13, 1), (21, 12, 1), (22, 10, 1), (23, 8, 1),  # F#m: G5 - F#5 - E5 - D5
    (24, 13, 2), (26, 17, 2),                          # G:   G5 - B5 (peak)
    (28, 15, 1), (29, 13, 1), (30, 10, 2),             # Em:  A5 - G5 - E5
]
# "himno": trailer anthem built on one catchy dotted hook ("ta-ta TAAA")
# repeated up the scale; the second half answers an octave higher.
EPIC_THEMES["himno"] = [
    (0, 7, 0.75), (0.75, 7, 0.25), (1, 12, 1), (2, 10, 0.75), (2.75, 8, 0.25), (3, 7, 1),       # C# C# F# E D C#
    (4, 3, 0.75), (4.75, 3, 0.25), (5, 7, 1), (6, 5, 1), (7, 3, 1),                             # A A C# B A
    (8, 8, 0.75), (8.75, 8, 0.25), (9, 13, 1), (10, 12, 0.75), (10.75, 10, 0.25), (11, 8, 1),   # D D G F# E D
    (12, 5, 0.75), (12.75, 5, 0.25), (13, 10, 1), (14, 8, 1), (15, 7, 1),                       # B B E D C#
    (16, 7, 0.75), (16.75, 7, 0.25), (17, 12, 1), (18, 15, 0.75), (18.75, 13, 0.25), (19, 12, 1),  # C# C# F# A G F#
    (20, 10, 0.75), (20.75, 10, 0.25), (21, 15, 1), (22, 13, 1), (23, 12, 1),                   # E E A G F#
    (24, 13, 0.75), (24.75, 13, 0.25), (25, 17, 1), (26, 15, 0.75), (26.75, 13, 0.25), (27, 17, 1),  # G G B A G B
    (28, 19, 2), (30, 17, 1), (31, 15, 1),                                                      # C#6 B A
]
# "villano": dark and mysterious, harmonic minor. Its signature is the
# F#-E#-F# turn (the E# leading tone) and chromatic tension before each bar.
EPIC_THEMES["villano"] = [
    (0, 12, 1), (1, 11, 0.5), (1.5, 12, 0.5), (2, 15, 1), (3, 13, 1),     # F# E# F# A G
    (4, 12, 1), (5, 11, 0.5), (5.5, 12, 0.5), (6, 7, 2),                  # F# E# F# C#
    (8, 13, 1), (9, 12, 0.5), (9.5, 13, 0.5), (10, 17, 1), (11, 15, 1),   # G F# G B A
    (12, 13, 1), (13, 10, 1), (14, 11, 2),                                # G E E# (tension)
    (16, 12, 1), (17, 11, 0.5), (17.5, 12, 0.5), (18, 19, 1), (19, 17, 1),  # F# E# F# C#6 B
    (20, 15, 1), (21, 13, 1), (22, 12, 2),                                # A G F#
    (24, 13, 1.5), (25.5, 12, 0.5), (26, 8, 1), (27, 7, 1),               # G F# D C#
    (28, 10, 1), (29, 11, 1), (30, 12, 2),                                # E E# F#
]
# "heroica": soaring arpeggios that leap higher each phrase, up to D6.
EPIC_THEMES["heroica"] = [
    (0, 0, 1), (1, 7, 1), (2, 15, 2),                     # F#4 C#5 A5 (rising arpeggio)
    (4, 13, 1), (5, 12, 1), (6, 10, 1.5), (7.5, 12, 0.5),  # G F# E F#
    (8, 8, 1), (9, 13, 1), (10, 17, 2),                   # D G B (G chord arpeggio)
    (12, 15, 1), (13, 13, 1), (14, 10, 2),                # A G E
    (16, 7, 1), (17, 12, 1), (18, 19, 2),                 # C#5 F#5 C#6 (higher)
    (20, 17, 1), (21, 15, 1), (22, 13, 1), (23, 12, 1),   # B A G F#
    (24, 13, 1), (25, 17, 1), (26, 20, 2),                # G B D6 (peak)
    (28, 19, 1), (29, 17, 1), (30, 15, 1), (31, 10, 1),   # C#6 B A E
]
THEME = "original"
EPIC_THEME = EPIC_THEMES[THEME]

# =============================================================================
# 2b) STYLE
# =============================================================================
# "trap":  half-time trap drums (clap on beat 3), trap hi-hat rolls, dark bell
#          melody + flute, no cowbell and no orchestra.
# "phonk": the drift phonk version with cowbell riffs and the orchestral layer.
STYLE = "trap"
# Instrument that plays the main trap riff: "koto", "bell", "piano", "guitar" or "pluck".
MELODY_SOUND = "koto"


def _steps(notes, length=64):
    out = [None] * length
    for k, s in notes:
        out[k] = s
    return out


# Trap bell riff: 4 bars with a dotted 3+3+2 bounce (steps 0, 3, 6), semitones
# from F#5. Bar 2 uses E# (the harmonic-minor leading tone) for a darker turn.
RIFF_TRAP = _steps([
    (0, 7), (3, 3), (6, 0), (8, 3), (10, 7), (12, 8), (14, 7),           # C# A F# A C# D C#
    (16, 5), (19, 3), (22, 0), (24, -1), (26, 0), (28, 3), (30, 0),      # B A F# E# F# A F#
    (32, 8), (35, 5), (38, 1), (40, 5), (42, 8), (44, 10), (46, 8),      # D B G B D E D
    (48, 7), (51, 5), (54, 3), (56, 1), (58, -2), (60, -1), (62, 0),     # C# B A G E E# F#
])
# Trap flute melody: 4 bars (start in beats, semitones from F#5, length in beats).
FLUTE_MELODY = [
    (0, 0, 1.5), (1.5, 3, 0.5), (2, 1, 1), (3, 0, 1),          # F# A G F#
    (4, -1, 1), (5, 0, 1), (6, -5, 2),                         # E# F# C#
    (8, 1, 1.5), (9.5, 3, 0.5), (10, 5, 1), (11, 3, 1),        # G A B A
    (12, 1, 1), (13, -2, 1), (14, -1, 2),                      # G E E#
]
KICK_TRAP_A = [0, 5, 10]
KICK_TRAP_B = [0, 3, 6, 10, 14]
TRAP_CLAP = 8                   # half-time: clap/snare on beat 3
# Hi-hat rolls per bar of the 4-bar cycle: (start step, length in steps,
# hits, pitch rise). Hats play 8ths (16ths in drop 2) everywhere else.
TRAP_HAT_ROLLS = [
    [(6, 2, 2, 1.0), (12, 4, 6, 1.15)],                  # 16ths, then a triplet roll
    [(10, 2, 4, 1.3), (14, 2, 2, 1.0)],                  # 32nd roll pitched up
    [(4, 2, 3, 1.1), (12, 2, 2, 1.0), (14, 2, 4, 1.25)],
    [(8, 4, 4, 1.0), (12, 4, 8, 1.5)],                   # fill: 32nds climbing in pitch
]
BREAK_SYNTH_LEAD = False        # the break now carries the orchestral theme instead

# 808 patterns per bar: (16th step, length in 16ths, semitones over the bar root)
BASS_A = [(0, 5, 0), (5, 3, 0), (8, 2, 0), (10, 4, 12), (14, 2, 0)]
BASS_B = [(0, 3, 0), (3, 3, 0), (6, 2, 0), (8, 2, 12), (10, 2, 0), (12, 2, 12), (14, 2, 7)]
KICK_A = [0, 5, 8, 10]
KICK_B = [0, 3, 6, 8, 10, 14]
SNARE_STEPS = [4, 12]
GLIDE = 0.06                    # 808 slide time between notes (s)
DROP_DIVE = 0.35                # 808 dive from one octave up at each drop (s)
BASS_DECAY = 1.4                # 808 sustain (s)
BASS_DRIVE = 9.0                # 808 distortion amount

COWBELL_DRIVE = 3.5

# =============================================================================
# 3) MIX
# =============================================================================
MIX = {
    "kick": 1.00, "snare": 1.20, "hats": 0.65, "sub": 0.85, "growl": 0.55,
    "cowbell": 1.00, "pad": 0.45, "drone": 0.30, "lead": 0.55, "fx": 0.45,
    "reverb": 0.55, "delay": 0.55,
    "strings": 0.60, "brass": 0.68, "choir": 0.50, "perc": 0.42, "flute": 0.50,
    "bell": 0.28,               # trap riff level per instrument (replaces the cowbell level)
    "piano": 0.30, "guitar": 1.10, "pluck": 0.63, "koto": 1.55,
}
REVERB_SEND = {"snare": 0.22, "cowbell": 0.16, "pad": 0.30, "lead": 0.40, "fx": 0.45, "hats": 0.04,
               "strings": 0.38, "brass": 0.28, "choir": 0.45, "perc": 0.25, "flute": 0.35}
SIDECHAIN_ORCH = (0.35, 0.15)   # orchestra breathes lightly with the kick
SIDECHAIN_808 = (0.5, 0.07)     # (depth, release s) - 808 ducks under the kick
SIDECHAIN_PAD = (0.7, 0.18)     # pad pumps with the kick
TARGET_RMS_DB = -10.5           # loudness of the drops before the limiter
CEILING_DB = -1.0               # final peak level


# =============================================================================
# DSP UTILITIES
# =============================================================================
rng = np.random.default_rng(SEED)


def secs(x):
    return int(round(x * SR))


def tt(n):
    return np.arange(n) / SR


def _sos(kind, fc, order):
    return signal.butter(order, fc, btype=kind, fs=SR, output="sos")


def lp(x, fc, order=2):
    return signal.sosfilt(_sos("lowpass", fc, order), x, axis=-1)


def hp(x, fc, order=2):
    return signal.sosfilt(_sos("highpass", fc, order), x, axis=-1)


def bp(x, lo, hi, order=2):
    return signal.sosfilt(_sos("bandpass", [lo, hi], order), x, axis=-1)


def norm(x, peak=1.0):
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


def fade_tail(x, sec):
    n = min(x.shape[-1], secs(sec))
    x = x.copy()
    x[..., x.shape[-1] - n:] *= np.linspace(1, 0, n)
    return x


def noise(n):
    return rng.standard_normal(n)


def db(x):
    return 20 * np.log10(max(x, 1e-12))


def blep_saw(freq, n, phase0=0.0):
    """Band-limited sawtooth (PolyBLEP). 'freq' may be a number or an array."""
    f = np.broadcast_to(np.asarray(freq, dtype=float), (n,))
    dt = f / SR
    ph = (phase0 + np.cumsum(dt)) % 1.0
    y = 2 * ph - 1
    m = ph < dt
    x = ph[m] / dt[m]
    y[m] -= x + x - x * x - 1
    m = ph > 1 - dt
    x = (ph[m] - 1) / dt[m]
    y[m] -= x * x + x + x + 1
    return y


def blep_square(freq, n, phase0=0.0):
    return 0.5 * (blep_saw(freq, n, phase0) - blep_saw(freq, n, phase0 + 0.5))


def resample(x, factor):
    """Plays a one-shot 'factor' times faster (higher pitch, shorter)."""
    idx = np.arange(0, len(x) - 1, factor)
    return np.interp(idx, np.arange(len(x)), x)


# =============================================================================
# INSTRUMENTS
# =============================================================================
def make_cowbell(semis):
    """808-style cowbell: two square waves (ratio 800/540) through a bandpass."""
    f = ROOT_COWBELL * 2 ** (semis / 12)
    n = secs(0.42)
    t = tt(n)
    tone = np.zeros(n)
    for f0, amp in ((f, 1.0), (f * COWBELL_RATIO, 0.8)):
        for k in range(1, 16, 2):                     # odd harmonics = square
            if f0 * k < SR * 0.45:
                tone += amp * np.sin(2 * np.pi * f0 * k * t) / k
    tone = bp(tone, 500, 7500)
    env = (0.7 * np.exp(-t / 0.022) + 0.3 * np.exp(-t / 0.16)) * np.minimum(1, t / 0.0008)
    return fade_tail(norm(np.tanh(COWBELL_DRIVE * norm(tone * env))), 0.03)


def make_kick():
    """Punchy kick tuned to F#1: fast pitch drop, click, hard saturation."""
    n = secs(0.6)
    t = tt(n)
    f = ROOT_808 + 190 * np.exp(-t / 0.03) + 900 * np.exp(-t / 0.0025)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * (0.85 * np.exp(-t / 0.28) + 0.15 * np.exp(-t / 0.05))
    click = hp(noise(n), 4000) * np.exp(-t / 0.004) * 0.35
    k = np.tanh(4.5 * (body + click)) * np.minimum(1, t / 0.0005)
    return fade_tail(norm(hp(k, 30, 1)), 0.06)


def make_snare():
    """Sharp snare: tonal body + noise + 3-burst clap + high crack."""
    n = secs(0.45)
    t = tt(n)
    body = np.sin(2 * np.pi * 195 * t) * np.exp(-t / 0.05) + 0.5 * np.sin(2 * np.pi * 290 * t) * np.exp(-t / 0.035)
    wires = bp(noise(n), 1800, 11000) * np.exp(-t / 0.09)
    clap_env = sum(np.where(t >= d, np.exp(-(t - d) / 0.006), 0) for d in (0, 0.011, 0.022))
    clap_env = clap_env + 0.6 * np.where(t >= 0.03, np.exp(-(t - 0.03) / 0.07), 0)
    clap = bp(noise(n), 900, 6000) * clap_env
    crack = hp(noise(n), 6000) * np.exp(-t / 0.004)
    s = 0.7 * norm(body) + 0.8 * norm(wires) + 0.6 * norm(clap) + 0.4 * norm(crack)
    return fade_tail(norm(np.tanh(2.2 * s)), 0.03)


def make_hat(decay, length):
    """Metallic hat: six detuned square waves (808 recipe) + noise, high-passed."""
    n = secs(length)
    t = tt(n)
    metal = sum(blep_square(f, n, rng.random()) for f in (205.3, 304.4, 369.6, 522.7, 540.0, 800.0))
    x = 0.6 * norm(metal) + 0.5 * norm(noise(n))
    x = hp(x, 7000, 4) * np.exp(-t / decay) * np.minimum(1, t / 0.0005)
    return fade_tail(norm(x), 0.005)


def make_impact():
    """Cinematic hit (stereo): sub boom diving to 28 Hz + thump + wide crash."""
    n = secs(3.0)
    t = tt(n)
    f = 28 + 55 * np.exp(-t / 0.25)
    boom = np.tanh(2.5 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.9))
    thump = lp(noise(n), 900) * np.exp(-t / 0.05)
    mono = norm(boom) + 0.5 * norm(thump)
    crash = np.stack([lp(hp(noise(n), 2500), 12000) * np.exp(-t / 0.8) for _ in range(2)])
    return fade_tail(norm(mono[None, :] + 0.45 * norm(crash)), 0.3)


def make_reverse_crash(dur):
    n = secs(dur)
    t = tt(n)
    x = np.stack([lp(hp(noise(n), 4000), 12000) * np.exp(-t / (dur / 3)) for _ in range(2)])
    return norm(x[:, ::-1])


def make_riser(dur):
    """Noise sweeping from dark to bright + rising saw, growing louder."""
    n = secs(dur)
    p = np.linspace(0, 1, n)
    amp = 0.02 + 0.98 * p ** 2.5
    out = np.zeros((2, n))
    for ch in range(2):
        nz = noise(n)
        out[ch] = ((1 - p) * lp(nz, 900) + p * hp(nz, 3500)) * amp
    sweep = hp(blep_saw(110 * 2 ** (3 * p), n), 150) * amp * 0.25
    out += sweep[None, :]
    return fade_tail(norm(out), 0.03)


def make_downlifter(dur):
    n = secs(dur)
    p = np.linspace(0, 1, n)
    out = np.zeros((2, n))
    for ch in range(2):
        nz = noise(n)
        out[ch] = (p * lp(nz, 700) + (1 - p) * hp(nz, 3000)) * (1 - p) ** 2
    return norm(out)


def make_pad_chord(semis_list, dur, release=1.0):
    """Dark supersaw chord, 3 detuned voices per note and per side (stereo)."""
    n = secs(dur + release)
    t = tt(n)
    out = np.zeros((2, n))
    detunes = ((-0.11, 0.0, 0.13), (-0.14, 0.02, 0.10))    # semitones, L / R
    for s in semis_list:
        f = ROOT_PAD * 2 ** (s / 12)
        for ch in range(2):
            for d in detunes[ch]:
                out[ch] += blep_saw(f * 2 ** (d / 12), n, rng.random())
    env = (1 - np.exp(-t / 0.25)) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.35))
    return out * env / (3 * len(semis_list))


def make_lead_note(semis, dur):
    """Dark lead: saw + sub-octave square, low-passed, delayed vibrato."""
    f = ROOT_LEAD * 2 ** (semis / 12)
    n = secs(dur + 0.4)
    t = tt(n)
    vib = 1 + 0.008 * np.sin(2 * np.pi * 5.2 * t) * np.clip((t - 0.25) / 0.3, 0, 1)
    x = 0.55 * blep_saw(f * vib, n) + 0.45 * blep_square(f * vib / 2, n)
    x = np.tanh(1.6 * lp(x, 2200))
    env = np.minimum(1, t / 0.015) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.12))
    return x * env


def make_drone(dur):
    """Low detuned saw rumble on F#2, swelling in and out."""
    n = secs(dur)
    p = np.linspace(0, 1, n)
    x = blep_saw(ROOT_808 * 2, n) + blep_saw(ROOT_808 * 2 * 1.004, n, 0.3) + 0.7 * blep_saw(ROOT_808, n, 0.6)
    return lp(x, 220) * np.sin(np.pi * p) ** 1.5


# ----------------------------------------------------------- orchestral layer
def _env(t, dur, attack, release):
    return (1 - np.exp(-t / attack)) * np.where(t < dur, 1.0, np.exp(-(t - dur) / release))


def make_strings(semis, dur, attack=0.12, release=0.45, bright=4500, tremolo=0.0, swell=False):
    """String section: 6 detuned saws (3 per side), delayed vibrato, soft attack."""
    f = ROOT_LEAD * 2 ** (semis / 12)
    n = secs(dur + 4 * release)
    t = tt(n)
    vib = 1 + 0.0045 * np.sin(2 * np.pi * 5.4 * t + rng.random() * 6.28) * np.clip((t - 0.25) / 0.4, 0, 1)
    out = np.zeros((2, n))
    for v in range(6):
        out[v % 2] += blep_saw(f * 2 ** ((v - 2.5) * 0.05 / 12) * vib, n, rng.random())
    out = hp(lp(out, min(bright, f * 12)), 140)
    env = _env(t, dur, attack, release)
    if tremolo:                                   # fast bowing, like a string tremolo
        env = env * (0.55 + 0.45 * np.abs(np.sin(np.pi * tremolo * t)))
    if swell:
        env = env * np.clip(t / dur, 0, 1) ** 2
    return out * env / 3


def make_brass(semis, dur, attack=0.03, release=0.18):
    """Brass: detuned saws with a pitch scoop and a filter that opens on attack."""
    f = ROOT_PAD * 2 ** (semis / 12)
    n = secs(dur + 5 * release)
    t = tt(n)
    scoop = 2 ** (-0.4 * np.exp(-t / 0.045) / 12)
    out = np.zeros((2, n))
    for v, d in enumerate((-0.07, 0.0, 0.06, 0.1)):
        out[v % 2] += blep_saw(f * 2 ** (d / 12) * scoop, n, rng.random())
    bright = np.clip(t / 0.09, 0, 1) * (0.7 + 0.3 * np.exp(-t / 0.5))
    y = np.tanh(1.6 * (bright * lp(out, 3800) + (1 - bright) * lp(out, 500)))
    return y * _env(t, dur, attack, release) / 2


def make_choir(semis_list, dur, attack=0.6, release=0.7):
    """'Aah' choir: detuned saws with vibrato through the formants of the vowel A."""
    n = secs(dur + 4 * release)
    t = tt(n)
    src = np.zeros((2, n))
    for s in semis_list:
        f = ROOT_PAD * 2 ** (s / 12)
        for v in range(4):
            vib = 1 + 0.005 * np.sin(2 * np.pi * (4.8 + 0.4 * v) * t + rng.random() * 6.28)
            src[v % 2] += blep_saw(f * 2 ** ((v - 1.5) * 0.07 / 12) * vib, n, rng.random())
    src = norm(src) + 0.04 * np.stack([hp(noise(n), 2500) for _ in range(2)])   # breath
    vowel = sum(g * bp(src, fc - bw, fc + bw) for fc, g, bw in ((800, 1.0, 80), (1150, 0.5, 90), (2900, 0.25, 120)))
    return norm(vowel * _env(t, dur, attack, release))


def make_taiko():
    n = secs(1.2)
    t = tt(n)
    f = 50 + 70 * np.exp(-t / 0.035)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.4)
    skin = lp(noise(n), 1500) * np.exp(-t / 0.025)
    return fade_tail(norm(hp(np.tanh(2 * (body + 0.4 * skin)), 35, 1)), 0.1)


def make_timpani(semis=0):
    """Timpani tuned to F#2 (+semis): inharmonic drum modes + mallet."""
    f0 = ROOT_808 * 2 * 2 ** (semis / 12)
    n = secs(2.5)
    t = tt(n)
    glide = 1 + 0.03 * np.exp(-t / 0.05)
    x = sum(g * np.sin(2 * np.pi * np.cumsum(f0 * r * glide) / SR) * np.exp(-t / d)
            for r, g, d in ((1.0, 1.0, 1.3), (1.504, 0.5, 0.7), (1.742, 0.3, 0.5), (2.0, 0.25, 0.4)))
    x = x + 0.3 * lp(noise(n), 800) * np.exp(-t / 0.01)
    return fade_tail(norm(x * np.minimum(1, t / 0.001)), 0.2)


def make_braam(dur=3.0):
    """Trailer 'braam': low brass cluster on F#, saturated, filter opening."""
    n = secs(dur)
    t = tt(n)
    out = np.zeros((2, n))
    for f, g in ((ROOT_808, 1.0), (ROOT_808 * 2, 0.8), (ROOT_808 * 3, 0.5), (ROOT_808 * 4, 0.4)):
        for ch, d in enumerate((-0.12, 0.1)):
            out[ch] += g * blep_saw(f * 2 ** (d / 12), n, rng.random())
    opening = np.clip(t / 0.25, 0, 1) * np.exp(-t / 1.2)
    y = np.tanh(2.5 * norm(opening * lp(out, 2500) + (1 - opening) * lp(out, 300)))
    y = hp(y, 70) * (1 - np.exp(-t / 0.05)) * np.exp(-t / 1.1)
    return fade_tail(norm(y), 0.3)


# ----------------------------------------------------------- trap melodies
def make_bell(semis):
    """Dark trap bell: FM synthesis with an inharmonic 3.5 ratio, long decay."""
    f = ROOT_COWBELL * 2 ** (semis / 12)
    n = secs(1.4)
    t = tt(n)
    index = 4 * np.exp(-t / 0.15)
    x = np.sin(2 * np.pi * f * t + index * np.sin(2 * np.pi * 3.5 * f * t))
    x += 0.3 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.3)
    x *= np.exp(-t / 0.6) * np.minimum(1, t / 0.001)
    return fade_tail(norm(np.tanh(1.3 * x)), 0.1)


def make_piano(semis):
    """Dark piano, one octave below the riff: inharmonic partials, two detuned strings, hammer."""
    f = ROOT_COWBELL / 2 * 2 ** (semis / 12)
    n = secs(1.8)
    t = tt(n)
    x = np.zeros(n)
    for k in range(1, 13):
        fk = k * f * np.sqrt(1 + 0.0004 * k * k)          # piano string stiffness
        if fk > SR * 0.45:
            break
        decay = 1.4 / k ** 0.7
        for d in (-0.4, 0.4):                              # two strings beating (cents)
            x += np.sin(2 * np.pi * fk * 2 ** (d / 1200) * t) * np.exp(-t / decay) / k ** 1.2
    x += 0.15 * lp(noise(n), 3000) * np.exp(-t / 0.006)   # hammer
    x = lp(x, 5000) * np.minimum(1, t / 0.002)
    return fade_tail(norm(x), 0.15)


def karplus(f, n, damping, bright, pluck_pos=0.0):
    """Plucked string (Karplus-Strong) tuned exactly with an allpass in the loop.

    damping: how long the string rings (closer to 1 = longer)
    bright:  low-pass of the pluck noise (Hz) - higher = brighter attack
    pluck_pos: 0..0.5, where the string is plucked (small = near the bridge, twangy)
    """
    total = SR / f - 0.5                  # loop = N + 0.5 (averaging) + D (allpass)
    N = int(np.floor(total - 0.1))
    D = total - N
    c = (1 - D) / (1 + D)
    burst = lp(noise(N), bright)
    if pluck_pos:                         # comb notch = pluck position on the string
        k = max(1, int(pluck_pos * N))
        burst[k:] -= burst[:-k].copy()
    exc = np.zeros(n)
    exc[:N] = burst
    a = np.zeros(N + 3)
    a[0], a[1] = 1.0, c
    a[N] -= damping / 2 * c
    a[N + 1] -= damping / 2 * (1 + c)
    a[N + 2] -= damping / 2
    return signal.lfilter([1.0, c], a, exc)


def make_guitar(semis):
    """Nylon-style plucked string, one octave below the riff."""
    f = ROOT_COWBELL / 2 * 2 ** (semis / 12)
    n = secs(1.6)
    y = karplus(f, n, damping=0.996, bright=4000)
    y = lp(hp(y, 90), 4500) * np.exp(-tt(n) / 0.9)
    return fade_tail(norm(y), 0.15)


def make_koto(semis):
    """Koto: bright string plucked near the bridge, shorter ring, a twangy attack
    and the small pressed-string waver ("oshide") koto players add after the pluck."""
    f = ROOT_COWBELL / 2 * 2 ** (semis / 12)
    n = secs(1.4)
    t = tt(n)
    y = karplus(f, n, damping=0.993, bright=9000, pluck_pos=0.12)
    y = hp(y, 120) + 0.25 * hp(y, 2500) * np.exp(-t / 0.04)          # twang
    y = y + 0.25 * bp(y, 380, 900)                                    # wooden body
    waver = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.18) / 0.25, 0, 1)
    y = np.interp(np.cumsum(waver) - waver[0], np.arange(n), y)       # ~ +/-7 cent waver
    y *= np.exp(-t / 0.7) * np.minimum(1, t / 0.0005)
    return fade_tail(norm(np.tanh(1.2 * norm(y))), 0.15)


def make_pluck(semis):
    """Synth pluck: two detuned saws, filter snapping shut, short decay."""
    f = ROOT_COWBELL * 2 ** (semis / 12)
    n = secs(0.9)
    t = tt(n)
    x = blep_saw(f * 2 ** (-0.08 / 12), n) + blep_saw(f * 2 ** (0.08 / 12), n, 0.37)
    snap = np.exp(-t / 0.06)
    x = snap * lp(x, 6500) + (1 - snap) * lp(x, 700)
    x *= np.exp(-t / 0.35) * np.minimum(1, t / 0.001)
    return fade_tail(norm(np.tanh(1.5 * x)), 0.08)


MELODY_SYNTHS = {"bell": None, "piano": make_piano, "guitar": make_guitar, "pluck": make_pluck, "koto": make_koto}


def make_flute(semis, dur):
    """Breathy trap flute: near-sine tone, delayed vibrato and breath noise."""
    f = ROOT_COWBELL * 2 ** (semis / 12)
    n = secs(dur + 0.6)
    t = tt(n)
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.0 * t) * np.clip((t - 0.15) / 0.3, 0, 1)
    ph = 2 * np.pi * np.cumsum(f * vib) / SR
    tone = np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)
    breath = bp(noise(n), f * 0.8, min(f * 4, SR * 0.45)) * (0.6 * np.exp(-t / 0.08) + 0.4)
    x = np.tanh(1.2 * (tone + 0.12 * norm(breath) * 3))
    return x * _env(t, dur, 0.06, 0.15)


def make_reverb_ir(seconds=3.5, decay=0.55):
    """Stereo hall: two decorrelated noise tails (wide), dark, with predelay."""
    n = secs(seconds)
    t = tt(n)
    pre = np.zeros(secs(0.02))
    irs = []
    for _ in range(2):
        ir = lp(noise(n) * np.exp(-t / decay), 7000)
        ir = np.concatenate([pre, ir])
        irs.append(ir / np.sqrt(np.sum(ir ** 2)))
    return irs


# =============================================================================
# TIMELINE
# =============================================================================
SECTION_START = {}
_b = 0
for _name, _bars in SECTIONS:
    SECTION_START[_name] = (_b, _bars)
    _b += _bars
TOTAL_BARS = _b
SONG_END = TOTAL_BARS * BAR
N = secs(SONG_END + TAIL)

BUSES = ["kick", "snare", "hats", "cowbell", "pad", "drone", "lead", "fx", "strings", "brass", "choir", "perc", "flute"]
bus = {name: np.zeros((2, N)) for name in BUSES}
kick_times = []
bass_notes = []                 # (start s, dur s, freq, glide s, from_freq or None, gain)


def bar_time(section, bar=0, step=0.0):
    return (SECTION_START[section][0] + bar) * BAR + step * STEP


def place(name, snd, t, gain=1.0, pan=0.0):
    """Adds a mono or stereo sound to a bus at time t (constant-power pan)."""
    i = secs(t)
    if i >= N or i < 0:
        return
    if snd.ndim == 1:
        a = (pan + 1) * np.pi / 4
        snd = np.stack([snd * np.cos(a), snd * np.sin(a)]) * np.sqrt(2)
    end = min(N, i + snd.shape[1])
    bus[name][:, i:end] += snd[:, : end - i] * gain


SND = {
    "kick": make_kick(), "snare": make_snare(),
    "hat": make_hat(0.018, 0.07), "ohat": make_hat(0.12, 0.35),
    "impact": make_impact(),
}
_cow_cache = {}


def cowbell(semis):
    """The main melodic one-shot: the cowbell (phonk) or the dark bell (trap)."""
    key = (STYLE, MELODY_SOUND, semis)
    if key not in _cow_cache:
        if STYLE == "trap":
            _cow_cache[key] = (MELODY_SYNTHS.get(MELODY_SOUND) or make_bell)(semis)
        else:
            _cow_cache[key] = make_cowbell(semis)
    return _cow_cache[key]


def is_gap(section, bar, step):
    """Last beat of a build (and of drop 2) is silent before the next hit."""
    last = SECTION_START[section][1] - 1
    return section in ("build1", "build2", "drop2") and bar == last and step >= 12


# ---------------------------------------------------------------- cowbell
def play_cowbell(section, riff, only_downbeats=False, gain=1.0, stop_bar=None):
    bars = SECTION_START[section][1] if stop_bar is None else stop_bar
    for b in range(bars):
        for k in range(16):
            if is_gap(section, b, k) or (only_downbeats and k % 4):
                continue
            semis = riff[(b * 16 + k) % len(riff)]
            if semis is None:
                continue
            accent = 1.0 if k % 4 == 0 else (0.82 if k % 2 == 0 else 0.68)
            place("cowbell", cowbell(semis), bar_time(section, b, k), gain * accent)


# ---------------------------------------------------------------- drums
def play_kick(section, b, k, gain=1.0):
    t = bar_time(section, b, k)
    place("kick", SND["kick"], t, gain)
    kick_times.append(t)


def drop_drums(section, kick_pattern, heavy):
    for b in range(SECTION_START[section][1]):
        cyc = b % 4
        for k in range(16):
            if is_gap(section, b, k):
                continue
            if k in kick_pattern:
                play_kick(section, b, k)
            if k in SNARE_STEPS or (heavy and cyc == 3 and k == 15):
                place("snare", SND["snare"], bar_time(section, b, k), 1.0 if k in SNARE_STEPS else 0.6)
            # hats: 16ths, 32nds on the last 16th of each beat (heavy), rolls
            if cyc == 1 and k >= 14:
                hits, span = 3, 1                     # triplet 32nds
            elif cyc == 3 and k >= 12:
                hits, span = (2, 1) if k < 15 else (4, 1)
            elif heavy and k % 4 == 3:
                hits, span = 2, 1
            else:
                hits, span = 1, 1
            for h in range(hits):
                vel = (1.0 if k % 2 == 0 else 0.62) if hits == 1 else 0.5 + 0.5 * h / (hits - 1)
                place("hats", SND["hat"], bar_time(section, b, k + h * span / hits), vel, pan=0.18 if h % 2 else -0.12)
            if k == 10 and cyc in (0, 2):
                place("hats", SND["ohat"], bar_time(section, b, k), 0.5, pan=0.25)


_hat_cache = {}


def hat_pitched(factor):
    key = round(factor, 3)
    if key not in _hat_cache:
        _hat_cache[key] = resample(SND["hat"], factor) if key != 1 else SND["hat"]
    return _hat_cache[key]


def trap_drums(section, kick_pattern, heavy):
    """Half-time trap groove: kicks with the 808, clap on beat 3, rolling hats."""
    for b in range(SECTION_START[section][1]):
        cyc = b % 4
        rolls = TRAP_HAT_ROLLS[cyc]
        covered = set()
        for start, span, hits, rise in rolls:
            covered.update(range(start, start + span))
            for h in range(hits):
                pos = start + h * span / hits
                if is_gap(section, b, int(pos)):
                    continue
                prog = h / max(1, hits - 1)
                place("hats", hat_pitched(1 + (rise - 1) * prog), bar_time(section, b, pos),
                      0.55 + 0.4 * prog, pan=0.2 if h % 2 else -0.15)
        for k in range(16):
            if is_gap(section, b, k):
                continue
            if k in kick_pattern:
                play_kick(section, b, k)
            if k == TRAP_CLAP:
                place("snare", SND["snare"], bar_time(section, b, k))
            if heavy and cyc == 3 and k == 15:
                place("snare", SND["snare"], bar_time(section, b, k), 0.45)
            if k not in covered and (k % 2 == 0 or heavy):
                place("hats", SND["hat"], bar_time(section, b, k), 0.9 if k % 4 == 0 else 0.6, pan=-0.1)
        if cyc in (0, 2) and 6 not in covered:
            place("hats", SND["ohat"], bar_time(section, b, 6), 0.45, pan=0.25)


def build_drums(section):
    bars = SECTION_START[section][1]
    total = bars * 16
    for b in range(bars):
        # snare roll: quarters -> 8ths -> 16ths -> 32nds, rising in pitch and level
        div = [4, 2, 1, 0.5][min(3, b)]
        pos = 0.0
        while pos < 16:
            k = int(pos)
            if is_gap(section, b, k):
                break
            prog = (b * 16 + pos) / total
            snd = resample(SND["snare"], 1 + 0.6 * prog)
            place("snare", snd, bar_time(section, b, pos), 0.3 + 0.7 * prog ** 1.5)
            pos += div
        for k in range(0, 16, 2 if b < 2 else 1):    # hats 8ths, then 16ths
            if not is_gap(section, b, k):
                place("hats", SND["hat"], bar_time(section, b, k), 0.5 if k % 4 else 0.8, pan=-0.1)
        if b < 2:
            for k in (0, 8):
                play_kick(section, b, k, 0.8)


# ---------------------------------------------------------------- 808
def add_808(section, pattern, dive=True, gain=1.0):
    for b in range(SECTION_START[section][1]):
        root = PROGRESSION[b % 4]
        for i, (k, length, off) in enumerate(pattern):
            if is_gap(section, b, k):
                continue
            f = ROOT_808 * 2 ** ((root + off) / 12)
            first = dive and b == 0 and i == 0
            bass_notes.append((bar_time(section, b, k), length * STEP, f,
                               DROP_DIVE if first else GLIDE, 2 * f if first else None, gain))


def render_808():
    freq = np.full(N, ROOT_808)
    amp = np.zeros(N)
    prev = None
    for start, dur, f, glide, f_from, gain in sorted(bass_notes):
        i0, i1 = secs(start), min(N, secs(start + dur))
        n = i1 - i0
        if n <= 0:
            continue
        t = tt(n)
        f0 = f_from or prev or f
        fr = f0 * (f / f0) ** np.clip(t / glide, 0, 1)
        freq[i0:i1] = fr * (1 + 0.25 * np.exp(-t / 0.01))     # small pitch punch
        env = np.exp(-t / BASS_DECAY) * np.minimum(1, t / 0.002)
        r = min(n, secs(0.015))
        env[n - r:] *= np.linspace(1, 0, r)
        amp[i0:i1] = env * gain
        prev = f
    s = np.sin(2 * np.pi * np.cumsum(freq) / SR) * amp
    sub = lp(s, 140)                                           # clean low end
    growl = np.tanh(BASS_DRIVE * s) + 0.3 * np.tanh(BASS_DRIVE * np.abs(s))
    growl = lp(hp(growl, 80), 3800)                            # audible harmonics
    return sub, growl


# =============================================================================
# ARRANGEMENT
# =============================================================================
def arrange():
    # ---- INTRO: opening hit, drone, pad, cowbell behind a closed filter
    place("fx", SND["impact"], 0.0, 0.6)
    place("drone", make_drone(SECTION_START["intro"][1] * BAR + 2.0), 0.0)
    trap = STYLE == "trap"
    riff_main = RIFF_TRAP if trap else RIFF_A
    riff_drop2 = RIFF_TRAP if trap else RIFF_B
    drums = trap_drums if trap else drop_drums
    play_cowbell("intro", riff_main)
    # ---- BUILDS
    for sec in ("build1", "build2"):
        bars = SECTION_START[sec][1]
        play_cowbell(sec, riff_main)
        build_drums(sec)
        place("fx", make_riser(bars * BAR - BEAT), bar_time(sec), 0.55)
        place("fx", make_reverse_crash(BAR), bar_time(sec, bars) - BAR, 0.5)
    # ---- DROP 1
    place("fx", SND["impact"], bar_time("drop1"), 1.0)
    play_cowbell("drop1", riff_main)
    drums("drop1", KICK_TRAP_A if trap else KICK_A, heavy=False)
    add_808("drop1", BASS_A)
    # ---- BREAK: downlifter, sub 808, lead, sparse cowbell
    place("fx", make_downlifter(2 * BAR), bar_time("break"), 0.5)
    place("fx", SND["impact"], bar_time("break"), 0.45)
    place("drone", make_drone(SECTION_START["break"][1] * BAR + 1.0), bar_time("break"))
    play_cowbell("break", riff_main, only_downbeats=True, gain=0.8)
    if trap:                     # flute melody twice over the break, light hats in the second half
        for rep_bar in (0, 4):
            for beat, semis, length in FLUTE_MELODY:
                place("flute", make_flute(semis, length * BEAT * 0.95), bar_time("break", rep_bar) + beat * BEAT)
        for b in range(4, 8):
            for k in range(0, 16, 2):
                place("hats", SND["hat"], bar_time("break", b, k), 0.5 if k % 4 else 0.7, pan=-0.1)
    for b in range(6):
        f = ROOT_808 * 2 ** (PROGRESSION[b % 4] / 12)
        bass_notes.append((bar_time("break", b), BAR, f, 0.12, None, 0.6))
    for b in range(0, 8, 2):
        play_kick("break", b, 0, 0.7)
    for beat, semis, length in (LEAD_MELODY if BREAK_SYNTH_LEAD else []):
        place("lead", make_lead_note(semis, length * BEAT), bar_time("break") + beat * BEAT)
    # ---- DROP 2
    place("fx", SND["impact"], bar_time("drop2"), 1.0)
    play_cowbell("drop2", riff_drop2)
    drums("drop2", KICK_TRAP_B if trap else KICK_B, heavy=True)
    if trap:                     # flute answers the bell in bars 1-4 and 9-12
        for rep_bar in (0, 8):
            for beat, semis, length in FLUTE_MELODY:
                place("flute", make_flute(semis, length * BEAT * 0.95), bar_time("drop2", rep_bar) + beat * BEAT, 0.8)
    add_808("drop2", BASS_B)
    place("fx", make_reverse_crash(BAR), bar_time("outro") - BAR, 0.5)
    # ---- OUTRO: cowbell fades in the filter, final hit + long 808
    place("fx", SND["impact"], bar_time("outro"), 0.7)
    play_cowbell("outro", riff_main, stop_bar=SECTION_START["outro"][1] - 1)
    place("drone", make_drone(SECTION_START["outro"][1] * BAR + 2.0), bar_time("outro"))
    last = bar_time("outro", SECTION_START["outro"][1] - 1)
    place("fx", SND["impact"], last, 1.0)
    play_kick("outro", SECTION_START["outro"][1] - 1, 0)
    bass_notes.append((last, 2.5, ROOT_808, DROP_DIVE, 2 * ROOT_808, 1.0))
    # ---- PAD: one chord per bar everywhere except the drop-2 gap
    for b in range(TOTAL_BARS):
        place("pad", make_pad_chord([s for s in PAD_CHORDS[b % 4]], BAR), b * BAR)


# ---------------------------------------------------------------- orchestra
_brass_cache = {}


def brass_note(semis, dur):
    key = (semis, round(dur, 4))
    if key not in _brass_cache:
        _brass_cache[key] = make_brass(semis, dur)
    return _brass_cache[key]


def strings_theme(section, start_bar=0, octaves=(0, 12), gain=1.0, stretch=1.0, max_beats=32):
    """Plays the epic theme on strings (one layer per octave)."""
    t0 = bar_time(section, start_bar)
    for beat, semis, length in EPIC_THEME:
        if beat >= max_beats:
            break
        for o in octaves:
            place("strings", make_strings(semis + o, length * BEAT * stretch * 0.98, bright=6000 if o > 0 else 4500),
                  t0 + beat * BEAT * stretch, gain * (0.7 if o > 0 else 1.0))


def brass_theme(section, start_bar=0, gain=1.0):
    t0 = bar_time(section, start_bar)
    for beat, semis, length in EPIC_THEME:
        place("brass", brass_note(semis, length * BEAT * 0.95), t0 + beat * BEAT, gain)


def brass_stabs(section, steps=(0, 10), gain=1.0):
    for b in range(SECTION_START[section][1]):
        chord = PAD_CHORDS[b % 4] + [PROGRESSION[b % 4] - 12]
        for k in steps:
            if is_gap(section, b, k):
                continue
            for s in chord:
                place("brass", brass_note(s, 0.75 * BEAT), bar_time(section, b, k), gain)


def choir_chords(section, gains):
    for b, g in enumerate(gains):
        place("choir", make_choir(PAD_CHORDS[b % 4], BAR), bar_time(section, b), g)


def arrange_epic():
    if STYLE != "phonk":
        return
    taiko, timp = make_taiko(), make_timpani()
    # INTRO: choir swells in; solo strings hint the theme in bars 5-8
    choir_chords("intro", np.linspace(0.3, 0.8, SECTION_START["intro"][1]))
    strings_theme("intro", 4, octaves=(0,), gain=0.6, max_beats=16)
    for b in (0, 4):
        place("perc", taiko, bar_time("intro", b), 0.9)
    # BUILDS: tremolo strings swell + timpani roll, then the braam at the drop
    for sec in ("build1", "build2"):
        bars = SECTION_START[sec][1]
        length = bars * BAR - BEAT
        for s in (0, 7, 12):                                   # F#4, C#5, F#5
            place("strings", make_strings(s, length, attack=0.3, release=0.05, tremolo=12, swell=True), bar_time(sec), 0.8)
        for b in range(bars - 2, bars):
            for k in range(0, 16):
                for half in (0, 0.5):                          # 32nd-note roll
                    if is_gap(sec, b, k):
                        continue
                    prog = ((b - bars + 2) * 16 + k + half) / 32
                    place("perc", timp, bar_time(sec, b, k + half), 0.15 + 0.6 * prog ** 1.5, pan=0.15 if half else -0.15)
    for sec in ("drop1", "drop2"):
        place("fx", make_braam(), bar_time(sec), 0.9)
    # DROP 1: brass stabs + taiko from the start, epic theme on strings in the second half
    brass_stabs("drop1", gain=0.55)
    for b in range(SECTION_START["drop1"][1]):
        place("perc", taiko, bar_time("drop1", b), 0.7)
    strings_theme("drop1", 8, gain=0.9)
    # BREAK: choir + theme at half speed on violins and cellos, timpani on the chords
    choir_chords("break", [0.9] * SECTION_START["break"][1])
    strings_theme("break", 0, octaves=(0, -12), gain=0.9, stretch=2.0, max_beats=16)
    for b in range(0, 8, 2):
        place("perc", make_timpani(PROGRESSION[b % 4]), bar_time("break", b), 0.8)
    # DROP 2 (climax): theme on strings and brass twice, choir, taiko on 1 and 3 with fills
    for start in (0, 8):
        strings_theme("drop2", start, gain=1.0)
        brass_theme("drop2", start, gain=0.9)
    choir_chords("drop2", [0.8] * SECTION_START["drop2"][1])
    for b in range(SECTION_START["drop2"][1]):
        steps = [0, 8] + ([12, 13, 14, 15] if b % 4 == 3 else [])
        for k in steps:
            if not is_gap("drop2", b, k):
                place("perc", taiko, bar_time("drop2", b, k), 0.8 if k in (0, 8) else 0.55)
    # OUTRO: choir fades, then the final braam + held brass chord + timpani
    choir_chords("outro", [0.7, 0.5, 0.3])
    last = bar_time("outro", SECTION_START["outro"][1] - 1)
    place("fx", make_braam(4.0), last, 1.0)
    for s in (0, 3, 7, -12):
        place("brass", make_brass(s, 2.5, release=0.6), last, 0.9)
    place("perc", timp, last, 1.0)


def automation(points):
    """Piecewise-linear curve from (time s, value) points, sampled per sample."""
    ts, vs = zip(*points)
    return np.interp(np.arange(N) / SR, ts, vs)


def filter_sweep(x, amount, dark_hz, bright_hz):
    """Crossfades a dark and a bright low-pass version: a moving filter."""
    return amount * lp(x, bright_hz) + (1 - amount) * lp(x, dark_hz)


def sidechain(depth, release):
    env = np.ones(N)
    for tk in kick_times:
        i = secs(tk)
        m = min(N, i + secs(release * 6))
        t = tt(m - i)
        g = 1 - depth * np.minimum(1, t / 0.004) * np.exp(-t / release)
        env[i:m] = np.minimum(env[i:m], g)
    return env


def ping_pong(x, delay, feedback, taps=4):
    mono = lp(x.mean(axis=0), 5000)
    d = secs(delay)
    out = np.zeros((2, N))
    for k in range(1, taps + 1):
        if k * d >= N:
            break
        out[k % 2, k * d:] += mono[: N - k * d] * feedback ** k
    return out


def tape_stop(x, t0, t1):
    """Slows the whole mix to a halt between t0 and t1, like a stopping deck."""
    i0, i1 = secs(t0), secs(t1)
    n = i1 - i0
    tau = np.arange(n) / n
    pos = i0 + np.cumsum((1 - tau) ** 1.6)
    seg = np.stack([np.interp(pos, np.arange(N), x[c]) for c in range(2)])
    seg *= np.clip((1 - tau) / 0.25, 0, 1)
    x[:, i0:i1] = seg
    return x


def limiter(x, ceiling, look=0.012):
    """Look-ahead peak limiter: gain never lets a peak pass the ceiling."""
    peak = np.max(np.abs(x), axis=0)
    g = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    L = secs(look)
    g = ndimage.minimum_filter1d(g, 2 * L + 1)
    g = ndimage.uniform_filter1d(g, L + 1)
    return x * g


def section_rms(x, section):
    a, n = SECTION_START[section]
    seg = x[:, secs(a * BAR): secs((a + n) * BAR)]
    return np.sqrt(np.mean(seg ** 2))


# =============================================================================
# MAIN
# =============================================================================
def build_mix():
    """Arranges, renders and mixes the track. Returns (stems, master mix)."""
    arrange()
    arrange_epic()
    sub, growl = render_808()

    # Automation (time s, value): cowbell filter and pad brightness per section
    t = {name: bar_time(name) for name, _ in SECTIONS}
    cow_open = automation([(0, 0.0), (t["build1"], 0.85), (t["drop1"], 1.0), (t["break"], 0.3),
                           (t["build2"], 0.35), (t["drop2"], 1.0), (t["outro"], 1.0), (SONG_END - BAR, 0.0), (SONG_END + TAIL, 0.0)])
    pad_open = automation([(0, 0.0), (t["build1"], 0.6), (t["drop1"], 0.9), (t["drop1"] + BAR, 0.45),
                           (t["break"], 0.9), (t["build2"], 0.6), (t["drop2"], 0.9), (t["drop2"] + BAR, 0.5),
                           (t["outro"], 0.5), (SONG_END, 0.0), (SONG_END + TAIL, 0.0)])
    pad_level = automation([(0, 0.4), (t["build1"], 1.0), (t["drop1"], 0.8), (t["break"], 1.2),
                            (t["drop2"], 0.8), (t["outro"], 1.0), (SONG_END, 0.0), (SONG_END + TAIL, 0.0)])

    cow = filter_sweep(bus["cowbell"], cow_open, 650, 9000)
    pad = filter_sweep(bus["pad"], pad_open, 500, 2600) * pad_level
    pad *= sidechain(*SIDECHAIN_PAD)
    duck = sidechain(*SIDECHAIN_808)
    bass = (MIX["sub"] * sub + MIX["growl"] * growl) * duck

    stems = {
        "kick": MIX["kick"] * bus["kick"], "snare": MIX["snare"] * bus["snare"], "hats": MIX["hats"] * bus["hats"],
        "bass": np.stack([bass, bass]), "cowbell": MIX[MELODY_SOUND if STYLE == "trap" else "cowbell"] * cow, "pad": MIX["pad"] * pad,
        "drone": MIX["drone"] * bus["drone"], "lead": MIX["lead"] * bus["lead"], "fx": MIX["fx"] * bus["fx"],
    }
    orch = sidechain(*SIDECHAIN_ORCH)
    for name in ("strings", "brass", "choir", "flute"):
        stems[name] = MIX[name] * bus[name] * orch
    stems["perc"] = MIX["perc"] * bus["perc"]
    # Shared stereo hall reverb (returns high-passed so the low end stays clean)
    send = sum(REVERB_SEND.get(k, 0) * v for k, v in stems.items())
    ir_l, ir_r = make_reverb_ir()
    verb = np.stack([signal.fftconvolve(send[0], ir_l)[:N], signal.fftconvolve(send[1], ir_r)[:N]])
    stems["reverb"] = MIX["reverb"] * hp(verb, 200)
    # Ping-pong delay (dotted 8th) on cowbell and lead
    stems["delay"] = MIX["delay"] * ping_pong(0.5 * stems["cowbell"] + stems["lead"] + 0.5 * stems["flute"], 3 * STEP, 0.38)

    mix = sum(stems.values())
    mix = tape_stop(mix, bar_time("break") - 2 * BEAT, bar_time("break"))

    # ---- Master: low cut, mono lows, gain to target, soft clip, limiter
    mix = hp(mix, 28)
    mid, side = (mix[0] + mix[1]) / 2, hp((mix[0] - mix[1]) / 2, 140)
    mix = np.stack([mid + side, mid - side])
    gain = 10 ** (TARGET_RMS_DB / 20) / section_rms(mix, "drop1")
    mix *= gain
    knee = 0.8
    a = np.abs(mix)
    mix = np.sign(mix) * np.where(a < knee, a, knee + (1 - knee) * np.tanh((a - knee) / (1 - knee)))
    ceiling = 10 ** (CEILING_DB / 20)
    mix = np.clip(limiter(mix, ceiling), -ceiling, ceiling)
    fade = secs(TAIL)
    mix[:, -fade:] *= np.linspace(1, 0, fade) ** 2
    return stems, mix


def main():
    global THEME, EPIC_THEME, STYLE
    for arg in sys.argv[1:]:
        if arg in ("trap", "phonk"):
            STYLE = arg
        elif arg in EPIC_THEMES:
            THEME, EPIC_THEME = arg, EPIC_THEMES[arg]
        else:
            sys.exit(f"Unknown option {arg!r}. Use trap or phonk, and for phonk a theme: {', '.join(EPIC_THEMES)}")
    out = "black_asphalt_trap.wav" if STYLE == "trap" else OUTPUT_FILE
    print(f"Style: {STYLE}" + (f"  |  epic theme: {THEME}" if STYLE == "phonk" else ""))
    _, mix = build_mix()
    wavfile.write(out, SR, (mix.T * 32767).astype(np.int16))

    print(f"Done: {out}  |  {BPM} BPM  |  {TOTAL_BARS} bars  |  {(SONG_END + TAIL):.1f} s  |  stereo")
    for name, (start, bars) in SECTION_START.items():
        m, s = divmod(start * BAR, 60)
        print(f"  {int(m)}:{s:05.2f}  {name:<7} {bars:>2} bars   RMS {db(section_rms(mix, name)):6.1f} dBFS")


if __name__ == "__main__":
    main()
