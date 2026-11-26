#!/usr/bin/env python3
"""
DESPUÉS DE LA LLUVIA - pieza tranquila para un edit de reflexión.

Síntesis desde cero (solo numpy + scipy): piano, cuerdas suaves, bajo cálido,
lluvia y crujido de vinilo. No usa samples.

    70 BPM · 4/4 · La menor -> Do mayor · estéreo · 44.1 kHz · ~1:30

Estructura (1 compás = 3.43 s):
    0:00   intro    acordes de piano en arpegio + lluvia
    0:13.7 tema A   melodía de piano, sencilla y con espacio
    0:41   tema B   la melodía sube, entran cuerdas y bajo (momento más emotivo)
    1:09   final    el piano se queda solo y termina en Do mayor

Versión "solo" (python reflexion.py solo): solo piano, más suave y sin que la
melodía suba tanto (se queda entre La4 y Sol5); sin cuerdas, bajo ni lluvia.

Uso:
    python reflexion.py        -> reflexion.wav
    python reflexion.py solo   -> reflexion_solo.wav
"""

import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

# =============================================================================
# 1) AJUSTES GENERALES
# =============================================================================
SR = 44100
BPM = 70                        # tempo lento (todo lo demás se calcula a partir de él)
SOLO = "solo" in sys.argv[1:]   # versión solo piano, más suave
OUTPUT_FILE = "reflexion_solo.wav" if SOLO else "reflexion.wav"
SEED = 7                        # misma semilla = mismo resultado
TAIL = 7.0                      # segundos para que el último acorde se apague

BEAT = 60.0 / BPM
BAR = BEAT * 4

# =============================================================================
# 2) ARMONÍA Y MELODÍA  (notas en MIDI: 60 = Do central, 69 = La 440 Hz)
# =============================================================================
# Acordes de la mano izquierda: graves a agudos. Colores suaves (add9, maj7).
CHORDS = {
    "Am": [45, 52, 57, 60, 64, 71],     # La2 Mi3 La3 Do4 Mi4 Si4   (La menor add9)
    "F":  [41, 48, 53, 57, 60, 64],     # Fa2 Do3 Fa3 La3 Do4 Mi4   (Fa maj7)
    "C":  [36, 43, 48, 52, 55, 62],     # Do2 Sol2 Do3 Mi3 Sol3 Re4 (Do add9)
    "G":  [43, 50, 55, 59, 62, 67],     # Sol2 Re3 Sol3 Si3 Re4 Sol4
}
LOOP = ["Am", "F", "C", "G"]            # vi - IV - I - V: melancólico y cálido

# Secciones: (nombre, acordes por compás)
SECTIONS = [
    ("intro", LOOP),
    ("tema_a", LOOP * 2),
    ("tema_b", LOOP * 2),
    ("final", ["Am", "F", "C", "C"]),
]

# Melodías: (inicio en tiempos dentro de la sección, nota MIDI, duración en tiempos)
# 72 = Do5, 74 = Re5, 76 = Mi5, 77 = Fa5, 79 = Sol5, 81 = La5, 83 = Si5, 84 = Do6
MELODY_A = [
    (0, 76, 1.5), (1.5, 74, 0.5), (2, 72, 2),           # Mi Re Do
    (4, 72, 1), (5, 74, 1), (6, 77, 2),                 # Do Re Fa
    (8, 76, 3), (11, 74, 1),                            # Mi . Re
    (12, 74, 2), (14, 71, 2),                           # Re Si
    (16, 76, 1.5), (17.5, 74, 0.5), (18, 72, 1), (19, 69, 1),   # Mi Re Do La
    (20, 72, 1), (21, 74, 1), (22, 77, 1), (23, 79, 1),         # Do Re Fa Sol
    (24, 79, 2), (26, 76, 2),                           # Sol Mi
    (28, 74, 4),                                        # Re (queda abierta)
]
MELODY_B = [
    (0, 81, 2), (2, 79, 1), (3, 76, 1),                 # La Sol Mi
    (4, 77, 2), (6, 76, 1), (7, 74, 1),                 # Fa Mi Re
    (8, 76, 2), (10, 79, 2),                            # Mi Sol
    (12, 79, 3), (15, 74, 1),                           # Sol . Re
    (16, 81, 1.5), (17.5, 83, 0.5), (18, 84, 2),        # La Si DO (punto más alto)
    (20, 81, 2), (22, 77, 2),                           # La Fa
    (24, 79, 1), (25, 76, 1), (26, 74, 1), (27, 72, 1), # Sol Mi Re Do
    (28, 71, 2), (30, 74, 2),                           # Si Re
]
# Versión solo: el tema B se queda en el mismo registro que el A (máximo Sol5)
MELODY_B_SOFT = [
    (0, 76, 2), (2, 74, 1), (3, 72, 1),                 # Mi Re Do
    (4, 72, 2), (6, 74, 1), (7, 77, 1),                 # Do Re Fa
    (8, 76, 3), (11, 72, 1),                            # Mi . Do
    (12, 74, 4),                                        # Re
    (16, 76, 1.5), (17.5, 77, 0.5), (18, 79, 2),        # Mi Fa Sol (punto más alto, suave)
    (20, 77, 2), (22, 76, 2),                           # Fa Mi
    (24, 74, 1), (25, 72, 1), (26, 71, 2),              # Re Do Si
    (28, 74, 4),                                        # Re
]
MELODY_FINAL_SOFT = [
    (0, 76, 2), (2, 74, 2),                             # Mi Re
    (4, 72, 4),                                         # Do
    (12, 72, 4),                                        # Do, sobre el último acorde
]
MELODY_FINAL = [
    (0, 76, 2), (2, 74, 2),                             # Mi Re
    (4, 72, 4),                                         # Do
    (12, 79, 4),                                        # Sol, sobre el último Do
]

# Patrón de arpegio (índices dentro del acorde, en corcheas)
ARP_LOW = [0, 1, 2, 3, 4, 3, 2, 1]
ARP_HIGH = [0, 2, 3, 4, 5, 4, 3, 2]

# =============================================================================
# 3) MEZCLA
# =============================================================================
MIX = {"piano": 0.70, "melody": 1.0, "strings": 0.60, "bass": 0.09, "rain": 0.14, "vinyl": 0.05}
REVERB_SEND = {"piano": 0.35, "melody": 0.40, "strings": 0.55, "bass": 0.0, "rain": 0.10, "vinyl": 0.0}
HUMANIZE_MS = 8                 # pequeñas variaciones de tiempo, como al tocar en vivo
TARGET_RMS_DB = -20.0 if SOLO else -17.0   # volumen medio del tema B (tranquilo, sin aplastar)
CEILING_DB = -1.0               # pico máximo final


# =============================================================================
# UTILIDADES DSP
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


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def blep_saw(freq, n, phase0=0.0):
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


# =============================================================================
# INSTRUMENTOS
# =============================================================================
_piano_cache = {}


def piano_note(midi, vel, dur):
    """Piano: parciales algo desafinados (rigidez de la cuerda), dos cuerdas que
    'laten', martillo suave. Tocar flojo (vel baja) suena más oscuro y redondo.
    'dur' es cuánto se mantiene la tecla/pedal; después la nota se apaga."""
    key = (midi, round(vel, 2), round(dur, 2))
    if key in _piano_cache:
        return _piano_cache[key]
    f = mtof(midi)
    n = secs(dur + 3.0)
    t = tt(n)
    base_decay = np.clip(5.0 * (220 / f) ** 0.5, 1.0, 7.0)   # graves suenan más tiempo
    x = np.zeros(n)
    for k in range(1, 15):
        fk = k * f * np.sqrt(1 + 0.0003 * k * k)
        if fk > 15000:
            break
        amp = k ** -1.3 * vel ** (0.18 * (k - 1))             # menos brillo si se toca suave
        tau = base_decay / k ** 0.6
        for cents in (-0.3, 0.3):
            x += amp * np.sin(2 * np.pi * fk * 2 ** (cents / 1200) * t + rng.random() * 6.28) * np.exp(-t / tau)
    x += 0.04 * vel * lp(rng.standard_normal(n), 1500 + 2500 * vel) * np.exp(-t / 0.004)   # martillo
    x = lp(x, 3500 + 7000 * vel)
    env = np.minimum(1, t / 0.002) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.35))
    y = x * env * vel * 0.25
    _piano_cache[key] = y
    return y


def strings_chord(notes, dur):
    """Cuerdas suaves (estéreo): sierras desafinadas, filtro oscuro, entrada lenta."""
    n = secs(dur + 2.5)
    t = tt(n)
    out = np.zeros((2, n))
    for m in notes:
        f = mtof(m)
        for v in range(4):
            vib = 1 + 0.003 * np.sin(2 * np.pi * (4.6 + 0.3 * v) * t + rng.random() * 6.28)
            out[v % 2] += blep_saw(f * 2 ** ((v - 1.5) * 0.06 / 12) * vib, n, rng.random())
    out = hp(lp(out, 1800), 150)
    env = (1 - np.exp(-t / 0.9)) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.8))
    return out * env / (4 * len(notes))


def bass_note(midi, dur):
    """Bajo cálido: seno con un poco de segundo armónico."""
    f = mtof(midi)
    n = secs(dur + 1.0)
    t = tt(n)
    x = np.sin(2 * np.pi * f * t) + 0.15 * np.sin(4 * np.pi * f * t)
    env = (1 - np.exp(-t / 0.05)) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.3))
    return x * env


def make_rain(n):
    """Lluvia (estéreo): rumor de fondo que respira + gotas sueltas."""
    t = tt(n)
    out = np.zeros((2, n))
    swell = 0.75 + 0.25 * np.sin(2 * np.pi * 0.07 * t)
    for ch in range(2):
        out[ch] = lp(bp(rng.standard_normal(n), 600, 7000), 5000) * swell
    drops = rng.random(n) < 22 / SR                           # ~22 gotas por segundo
    for ch in range(2):
        d = np.where(drops & (rng.random(n) < 0.5), rng.uniform(0.5, 1.5, n), 0.0)
        out[ch] += 3 * bp(d, 2000, 9000)
    return out / np.max(np.abs(out))


def make_vinyl(n):
    """Crujido de vinilo muy suave."""
    clicks = np.where(rng.random(n) < 6 / SR, rng.uniform(-1, 1, n), 0.0)
    hiss = 0.05 * hp(rng.standard_normal(n), 3000)
    x = hp(clicks, 1500) + hiss
    return np.stack([x, np.roll(x, 37)]) / np.max(np.abs(x))


def make_reverb_ir(seconds=4.0, decay=0.8):
    n = secs(seconds)
    t = tt(n)
    pre = np.zeros(secs(0.03))
    irs = []
    for _ in range(2):
        ir = lp(rng.standard_normal(n) * np.exp(-t / decay), 6000)
        ir = np.concatenate([pre, ir])
        irs.append(ir / np.sqrt(np.sum(ir ** 2)))
    return irs


# =============================================================================
# LÍNEA DE TIEMPO
# =============================================================================
SECTION_START = {}
_b = 0
for _name, _chords in SECTIONS:
    SECTION_START[_name] = _b
    _b += len(_chords)
TOTAL_BARS = _b
SONG_END = TOTAL_BARS * BAR
N = secs(SONG_END + TAIL)
bus = {k: np.zeros((2, N)) for k in MIX}


def place(name, snd, t, gain=1.0, pan=0.0):
    i = secs(t)
    if i >= N:
        return
    i = max(0, i)
    if snd.ndim == 1:
        a = (pan + 1) * np.pi / 4
        snd = np.stack([snd * np.cos(a), snd * np.sin(a)]) * np.sqrt(2)
    end = min(N, i + snd.shape[1])
    bus[name][:, i:end] += snd[:, : end - i] * gain


def human():
    return rng.normal(0, HUMANIZE_MS / 1000)


def pitch_pan(midi):
    """Como en un piano real: graves un poco a la izquierda, agudos a la derecha."""
    return float(np.clip((midi - 60) / 40, -0.35, 0.35))


# =============================================================================
# ARREGLO
# =============================================================================
def play_arpeggios(section, gain, high=False):
    start = SECTION_START[section]
    chords = dict(SECTIONS)[section]
    pattern = ARP_HIGH if high else ARP_LOW
    for b, name in enumerate(chords):
        chord = CHORDS[name]
        t0 = (start + b) * BAR
        for i, idx in enumerate(pattern):
            beat = i * 0.5
            vel = (0.42 if i == 0 else 0.30) * rng.uniform(0.9, 1.08)
            dur = BAR - beat * BEAT                            # pedal hasta el final del compás
            m = chord[idx]
            place("piano", piano_note(m, round(vel, 2), dur), t0 + beat * BEAT + human(), gain, pitch_pan(m))


def play_melody(section, melody, gain=1.0):
    t0 = SECTION_START[section] * BAR
    soft = 0.8 if SOLO else 1.0                                # toque más flojo en la versión solo
    for beat, m, length in melody:
        phrase = soft * (0.62 + 0.12 * np.sin(np.pi * (beat % 8) / 8))  # cada frase crece y se relaja
        vel = round(phrase * rng.uniform(0.94, 1.04), 2)
        # un poco por detrás del pulso: suena relajado
        place("melody", piano_note(m, vel, length * BEAT * 1.05), t0 + beat * BEAT + 0.012 + human(), gain, 0.12)


def arrange_solo():
    """Solo piano: arpegios cerrados y melodía en un registro medio, todo más suave."""
    play_arpeggios("intro", 0.8)
    play_arpeggios("tema_a", 0.7)
    play_melody("tema_a", MELODY_A)
    play_arpeggios("tema_b", 0.7)
    play_melody("tema_b", MELODY_B_SOFT)
    chords = dict(SECTIONS)["final"]
    for b, name in enumerate(chords[:2]):
        t = (SECTION_START["final"] + b) * BAR
        for i, idx in enumerate(ARP_LOW):
            m = CHORDS[name][idx]
            place("piano", piano_note(m, 0.27 if i else 0.36, BAR - i * 0.5 * BEAT), t + i * 0.5 * BEAT + human(), 0.7, pitch_pan(m))
    t_last = (SECTION_START["final"] + 2) * BAR
    for i, m in enumerate(CHORDS["C"]):                       # acorde roto lento, que se apaga solo
        place("piano", piano_note(m, 0.33, 2 * BAR + TAIL - 1.5), t_last + i * 0.11, 0.8, pitch_pan(m))
    play_melody("final", MELODY_FINAL_SOFT, 0.9)


def arrange():
    if SOLO:
        return arrange_solo()
    # INTRO: arpegios solos
    play_arpeggios("intro", 0.9)
    # TEMA A: melodía sobre los arpegios
    play_arpeggios("tema_a", 0.8)
    play_melody("tema_a", MELODY_A)
    # TEMA B: melodía más alta, arpegios abiertos, cuerdas y bajo
    play_arpeggios("tema_b", 0.75, high=True)
    play_melody("tema_b", MELODY_B, 1.05)
    for b, name in enumerate(dict(SECTIONS)["tema_b"]):
        t = (SECTION_START["tema_b"] + b) * BAR
        place("strings", strings_chord(CHORDS[name][2:5], BAR), t)
        place("bass", bass_note(CHORDS[name][0], BAR * 0.95), t)
    # FINAL: piano casi solo; acorde final arpegiado lentamente y dejado sonar
    chords = dict(SECTIONS)["final"]
    for b, name in enumerate(chords[:2]):
        t = (SECTION_START["final"] + b) * BAR
        place("strings", strings_chord(CHORDS[name][2:5], BAR), t, 0.6)
        for i, idx in enumerate(ARP_LOW):
            m = CHORDS[name][idx]
            place("piano", piano_note(m, 0.3 if i else 0.4, BAR - i * 0.5 * BEAT), t + i * 0.5 * BEAT + human(), 0.75, pitch_pan(m))
    t_last = (SECTION_START["final"] + 2) * BAR
    for i, m in enumerate(CHORDS["C"]):                       # acorde roto lento
        place("piano", piano_note(m, 0.38, 2 * BAR + TAIL - 1.5), t_last + i * 0.09, 0.85, pitch_pan(m))
    play_melody("final", MELODY_FINAL, 0.95)
    # Ambiente: lluvia y vinilo durante todo el tema
    place("rain", make_rain(N), 0.0)
    place("vinyl", make_vinyl(N), 0.0)


def automation(points):
    ts, vs = zip(*points)
    return np.interp(np.arange(N) / SR, ts, vs)


# =============================================================================
# MEZCLA Y MASTER
# =============================================================================
def build_mix():
    arrange()
    end = SONG_END + TAIL
    rain_level = automation([(0, 0.0), (3, 1.0), (SECTION_START["tema_b"] * BAR, 0.8), (SONG_END, 1.0), (end, 0.0)])
    stems = {k: MIX[k] * bus[k] for k in MIX}
    stems["rain"] = stems["rain"] * rain_level
    stems["vinyl"] = stems["vinyl"] * rain_level
    send = sum(REVERB_SEND[k] * v for k, v in stems.items())
    ir_l, ir_r = make_reverb_ir()
    verb = np.stack([signal.fftconvolve(send[0], ir_l)[:N], signal.fftconvolve(send[1], ir_r)[:N]])
    stems["reverb"] = 0.8 * hp(verb, 180)
    mix = sum(stems.values())

    mix = hp(mix, 30)
    seg = mix[:, secs(SECTION_START["tema_b"] * BAR): secs(SECTION_START["final"] * BAR)]
    mix *= 10 ** (TARGET_RMS_DB / 20) / np.sqrt(np.mean(seg ** 2))
    knee = 0.85                                                # solo redondea picos sueltos
    a = np.abs(mix)
    mix = np.sign(mix) * np.where(a < knee, a, knee + (1 - knee) * np.tanh((a - knee) / (1 - knee)))
    ceiling = 10 ** (CEILING_DB / 20)
    if np.max(np.abs(mix)) > ceiling:
        mix *= ceiling / np.max(np.abs(mix))
    fade = secs(3.0)
    mix[:, -fade:] *= np.linspace(1, 0, fade) ** 2
    return stems, mix


def main():
    _, mix = build_mix()
    wavfile.write(OUTPUT_FILE, SR, (mix.T * 32767).astype(np.int16))
    print(f"Listo: {OUTPUT_FILE}  |  {BPM} BPM  |  {TOTAL_BARS} compases  |  {SONG_END + TAIL:.1f} s  |  estéreo")
    for name, chords in SECTIONS:
        a = SECTION_START[name] * BAR
        seg = mix[:, secs(a): secs(a + len(chords) * BAR)]
        m, s = divmod(a, 60)
        print(f"  {int(m)}:{s:05.2f}  {name:<7} {len(chords):>2} compases   RMS {20 * np.log10(np.sqrt(np.mean(seg ** 2))):6.1f} dBFS")


if __name__ == "__main__":
    main()
