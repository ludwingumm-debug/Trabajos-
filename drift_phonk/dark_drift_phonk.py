#!/usr/bin/env python3
"""
BLACK ASPHALT - an original dark drift phonk track, synthesized from scratch.

Only numpy + scipy: every sound (cowbell, 808, kick, snare, hats, pads, lead,
risers, impacts) is generated here; no samples are used.

    150 BPM · 4/4 · key of F# minor (with the dark phrygian G) · stereo · 44.1 kHz

Arrangement (60 bars, about 1:40):
    intro   8 bars  filtered cowbell, dark pad, drone, opening impact
    build   4 bars  snare roll, riser, reverse crash, one beat of silence
    drop 1 16 bars  full beat: deep 808 with dive, kick, snare, hats, cowbell riff A
    break   8 bars  tape stop, dark lead melody, sub 808, filtered cowbell
    build   4 bars  second build
    drop 2 16 bars  heavier 808 and kick patterns, higher cowbell riff B
    outro   4 bars  cowbell fades in the filter, final impact

Usage:
    python dark_drift_phonk.py          -> writes dark_drift_phonk.wav
"""

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
}
REVERB_SEND = {"snare": 0.22, "cowbell": 0.16, "pad": 0.30, "lead": 0.40, "fx": 0.45, "hats": 0.04}
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

BUSES = ["kick", "snare", "hats", "cowbell", "pad", "drone", "lead", "fx"]
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
    if semis not in _cow_cache:
        _cow_cache[semis] = make_cowbell(semis)
    return _cow_cache[semis]


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
    play_cowbell("intro", RIFF_A)
    # ---- BUILDS
    for sec in ("build1", "build2"):
        bars = SECTION_START[sec][1]
        play_cowbell(sec, RIFF_A)
        build_drums(sec)
        place("fx", make_riser(bars * BAR - BEAT), bar_time(sec), 0.55)
        place("fx", make_reverse_crash(BAR), bar_time(sec, bars) - BAR, 0.5)
    # ---- DROP 1
    place("fx", SND["impact"], bar_time("drop1"), 1.0)
    play_cowbell("drop1", RIFF_A)
    drop_drums("drop1", KICK_A, heavy=False)
    add_808("drop1", BASS_A)
    # ---- BREAK: downlifter, sub 808, lead, sparse cowbell
    place("fx", make_downlifter(2 * BAR), bar_time("break"), 0.5)
    place("fx", SND["impact"], bar_time("break"), 0.45)
    place("drone", make_drone(SECTION_START["break"][1] * BAR + 1.0), bar_time("break"))
    play_cowbell("break", RIFF_A, only_downbeats=True, gain=0.8)
    for b in range(6):
        f = ROOT_808 * 2 ** (PROGRESSION[b % 4] / 12)
        bass_notes.append((bar_time("break", b), BAR, f, 0.12, None, 0.6))
    for b in range(0, 8, 2):
        play_kick("break", b, 0, 0.7)
    for beat, semis, length in LEAD_MELODY:
        place("lead", make_lead_note(semis, length * BEAT), bar_time("break") + beat * BEAT)
    # ---- DROP 2
    place("fx", SND["impact"], bar_time("drop2"), 1.0)
    play_cowbell("drop2", RIFF_B)
    drop_drums("drop2", KICK_B, heavy=True)
    add_808("drop2", BASS_B)
    place("fx", make_reverse_crash(BAR), bar_time("outro") - BAR, 0.5)
    # ---- OUTRO: cowbell fades in the filter, final hit + long 808
    place("fx", SND["impact"], bar_time("outro"), 0.7)
    play_cowbell("outro", RIFF_A, stop_bar=SECTION_START["outro"][1] - 1)
    place("drone", make_drone(SECTION_START["outro"][1] * BAR + 2.0), bar_time("outro"))
    last = bar_time("outro", SECTION_START["outro"][1] - 1)
    place("fx", SND["impact"], last, 1.0)
    play_kick("outro", SECTION_START["outro"][1] - 1, 0)
    bass_notes.append((last, 2.5, ROOT_808, DROP_DIVE, 2 * ROOT_808, 1.0))
    # ---- PAD: one chord per bar everywhere except the drop-2 gap
    for b in range(TOTAL_BARS):
        place("pad", make_pad_chord([s for s in PAD_CHORDS[b % 4]], BAR), b * BAR)


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
        "bass": np.stack([bass, bass]), "cowbell": MIX["cowbell"] * cow, "pad": MIX["pad"] * pad,
        "drone": MIX["drone"] * bus["drone"], "lead": MIX["lead"] * bus["lead"], "fx": MIX["fx"] * bus["fx"],
    }
    # Shared stereo hall reverb (returns high-passed so the low end stays clean)
    send = sum(REVERB_SEND.get(k, 0) * v for k, v in stems.items())
    ir_l, ir_r = make_reverb_ir()
    verb = np.stack([signal.fftconvolve(send[0], ir_l)[:N], signal.fftconvolve(send[1], ir_r)[:N]])
    stems["reverb"] = MIX["reverb"] * hp(verb, 200)
    # Ping-pong delay (dotted 8th) on cowbell and lead
    stems["delay"] = MIX["delay"] * ping_pong(0.5 * stems["cowbell"] + stems["lead"], 3 * STEP, 0.38)

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
    _, mix = build_mix()
    wavfile.write(OUTPUT_FILE, SR, (mix.T * 32767).astype(np.int16))

    print(f"Done: {OUTPUT_FILE}  |  {BPM} BPM  |  {TOTAL_BARS} bars  |  {(SONG_END + TAIL):.1f} s  |  stereo")
    for name, (start, bars) in SECTION_START.items():
        m, s = divmod(start * BAR, 60)
        print(f"  {int(m)}:{s:05.2f}  {name:<7} {bars:>2} bars   RMS {db(section_rms(mix, name)):6.1f} dBFS")


if __name__ == "__main__":
    main()
