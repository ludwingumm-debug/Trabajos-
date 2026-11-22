#!/usr/bin/env python3
"""
Generador de beat de DRIFT PHONK - síntesis 100% desde cero (solo numpy + scipy).

Produce un archivo WAV con:
  - Cencerro (cowbell) de dos tonos tocando una melodía en semicorcheas (escala menor)
  - Bajo 808 distorsionado con glide (deslizamiento de tono) siguiendo la tónica
  - Bombo fuerte, caja con reverb y hi-hats rápidos con redobles
  - Saturación suave en el master + normalización

Estructura: INTRO_BARS compases solo con cencerro y luego entra todo el beat.

Uso:
    python phonk_beat.py
"""

import math

import numpy as np
from scipy import signal
from scipy.io import wavfile

# =============================================================================
# 1) PARÁMETROS GENERALES  (cambia aquí tempo, duración, nombre del archivo...)
# =============================================================================
SR = 44100                  # Frecuencia de muestreo (Hz)
BPM = 130                   # Tempo en pulsos por minuto
DURATION = 30.0             # Duración total del beat en segundos
INTRO_BARS = 4              # Compases de intro (solo cencerro)
STEPS_PER_BAR = 16          # 16 semicorcheas por compás (4/4)
OUTPUT_FILE = "phonk_beat.wav"
SEED = 7                    # Semilla del ruido (mismo valor = mismo sonido)

# Duraciones derivadas del tempo (no hace falta tocarlas)
BEAT = 60.0 / BPM           # Duración de una negra
STEP = BEAT / 4             # Duración de una semicorchea
BAR = BEAT * 4              # Duración de un compás de 4/4

# =============================================================================
# 2) TONALIDAD Y NOTAS
# =============================================================================
# Frecuencias base de los dos osciladores del cencerro (estilo 808).
COWBELL_F1 = 540.0
COWBELL_F2 = 800.0

# La tónica del 808 se deriva del cencerro (3 octavas abajo) para que todo
# esté afinado entre sí. 540 / 8 = 67.5 Hz (aprox. un C#2 algo bajo).
TONIC_HZ = COWBELL_F1 / 8

# Escala menor natural en semitonos: grados 0..6 -> 1ª, 2ª, b3ª, 4ª, 5ª, b6ª, b7ª
# (prueba [0, 2, 3, 5, 7, 8, 11] para menor armónica, muy típica en phonk)
SCALE = [0, 2, 3, 5, 7, 8, 10]

# Melodía del cencerro: un valor por semicorchea, se repite en bucle.
# Cada número es un GRADO de la escala (0 = tónica, 4 = quinta, 7 = tónica una
# octava arriba, -1 = séptima una octava abajo...). None = silencio.
# Aquí hay 2 compases (32 semicorcheas); puedes poner la longitud que quieras.
COWBELL_MELODY = [
    0, None, 0, 2,    0, None, 4, None,   3, None, 3, 2,    0, None, -1, None,
    0, None, 0, 2,    0, None, 4, 5,      4, None, 3, 2,    3, 2, 0, None,
]

# Progresión del 808: un grado de la escala por compás (se repite en bucle).
# 0 = i, 5 = bVI, 4 = v  ->  i - i - bVI - v
BASS_PROGRESSION = [0, 0, 5, 4]

# Ritmo del 808 dentro de cada compás:
#   (semicorchea de inicio, duración en semicorcheas, desplazamiento en semitonos)
# El salto de +12 (una octava) con glide es el "slide" clásico del phonk.
BASS_PATTERN = [
    (0, 6, 0),
    (6, 4, 0),
    (10, 3, 12),
    (13, 3, 0),
]
GLIDE_TIME = 0.07           # Segundos que tarda el 808 en deslizarse a la nota nueva
BASS_DECAY = 0.9            # Caída del volumen del 808 (segundos, más = más largo)
BASS_DRIVE = 4.0            # Distorsión del 808 (1 = limpio, 8 = muy sucio)

# =============================================================================
# 3) PATRONES DE BATERÍA (posiciones en semicorcheas, 0..15)
# =============================================================================
KICK_STEPS = [0, 6, 10]     # Bombo
SNARE_STEPS = [4, 12]       # Caja en los tiempos 2 y 4

# Redobles de hi-hat: {compás dentro de un ciclo de 4: [(inicio, golpes, duración)]}
#   inicio   -> semicorchea donde empieza el redoble
#   golpes   -> número de golpes del redoble
#   duración -> cuántas semicorcheas ocupa
# Ej: (14, 6, 2) = 6 golpes en 2 semicorcheas (fusas en tresillo).
HAT_ROLLS = {
    1: [(14, 6, 2)],
    3: [(8, 3, 1), (12, 8, 4)],
}

# =============================================================================
# 4) MEZCLA
# =============================================================================
VOLUMES = {
    "cowbell": 0.85,
    "kick": 0.80,
    "bass": 0.40,
    "snare": 1.00,
    "hats": 0.40,
}
COWBELL_REVERB_WET = 0.20   # Cantidad de reverb en el cencerro (0..1)
SNARE_REVERB_WET = 0.40     # Cantidad de reverb en la caja (0..1)
MASTER_DRIVE = 1.8          # Saturación del master (1 = suave, 3 = fuerte)
PEAK_DB = -1.0              # Pico máximo final en dBFS (evita clipping)
FADE_OUT = 1.0              # Fundido de salida en segundos


# =============================================================================
# UTILIDADES DSP
# =============================================================================
rng = np.random.default_rng(SEED)


def t_axis(seconds):
    """Vector de tiempo de 'seconds' segundos."""
    return np.arange(int(seconds * SR)) / SR


def _sos(kind, cutoff, order=2):
    return signal.butter(order, cutoff, btype=kind, fs=SR, output="sos")


def lowpass(x, fc, order=2):
    return signal.sosfilt(_sos("lowpass", fc, order), x)


def highpass(x, fc, order=2):
    return signal.sosfilt(_sos("highpass", fc, order), x)


def bandpass(x, lo, hi, order=2):
    return signal.sosfilt(_sos("bandpass", [lo, hi], order), x)


def normalize(x, peak=1.0):
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


def attack_ramp(n, seconds):
    """Rampa lineal 0 -> 1 de 'seconds' segundos (evita clicks)."""
    env = np.ones(n)
    a = min(n, max(1, int(seconds * SR)))
    env[:a] = np.linspace(0, 1, a)
    return env


def fade_tail(x, seconds=0.02):
    """Fundido al final de un one-shot para que no haga 'click' al cortarse."""
    n = min(len(x), int(seconds * SR))
    x = x.copy()
    x[len(x) - n:] *= np.linspace(1, 0, n)
    return x


def add_at(buf, sound, time_sec, gain=1.0):
    """Suma 'sound' dentro de 'buf' empezando en 'time_sec' (recorta al final)."""
    i = int(round(time_sec * SR))
    if i >= len(buf):
        return
    end = min(len(buf), i + len(sound))
    buf[i:end] += sound[: end - i] * gain


def degree_to_semitones(degree):
    """Convierte un grado de la escala (puede pasar de octava) a semitonos."""
    octave, idx = divmod(degree, len(SCALE))
    return SCALE[idx] + 12 * octave


def make_reverb_ir(seconds, decay, predelay=0.015, brightness=6000):
    """Respuesta al impulso sintética: ruido con caída exponencial."""
    t = t_axis(seconds)
    ir = rng.standard_normal(len(t)) * np.exp(-t / decay)
    ir = lowpass(ir, brightness)
    ir = np.concatenate([np.zeros(int(predelay * SR)), ir])
    return ir / np.sqrt(np.sum(ir ** 2))  # normaliza la energía


def reverb(x, ir, wet):
    wet_sig = signal.fftconvolve(x, ir)[: len(x)]
    return x * (1 - wet) + wet_sig * wet


# =============================================================================
# INSTRUMENTOS (one-shots)
# =============================================================================
def make_cowbell(semitones, length=0.35):
    """Cencerro estilo 808: dos ondas cuadradas (540 y 800 Hz) + filtro pasabanda.

    Las ondas cuadradas se construyen sumando armónicos impares (sin aliasing).
    'semitones' transpone ambos tonos para tocar la melodía.
    """
    ratio = 2 ** (semitones / 12)
    t = t_axis(length)
    tone = np.zeros(len(t))
    for f0 in (COWBELL_F1 * ratio, COWBELL_F2 * ratio):
        for k in range(1, 16, 2):
            if f0 * k < SR / 2:
                tone += np.sin(2 * np.pi * f0 * k * t) / k
    tone = bandpass(tone, 450, 5000)
    # Ataque muy corto + caída rápida con una cola breve (carácter "metálico")
    env = (0.75 * np.exp(-t / 0.018) + 0.25 * np.exp(-t / 0.13)) * attack_ramp(len(t), 0.001)
    return fade_tail(normalize(np.tanh(2.0 * normalize(tone * env))))


def make_kick(length=0.55):
    """Bombo: seno con barrido de tono descendente + click + saturación."""
    t = t_axis(length)
    freq = 45 + 120 * np.exp(-t / 0.035)                 # 165 Hz -> 45 Hz
    body = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t / 0.2)
    click = highpass(rng.standard_normal(len(t)), 3000) * np.exp(-t / 0.003) * 0.25
    kick = np.tanh(3.5 * (body + click)) * attack_ramp(len(t), 0.0005)
    return fade_tail(normalize(kick), 0.05)


def make_snare(length=0.4):
    """Caja: ruido filtrado (bordonera) + dos tonos cortos (cuerpo)."""
    t = t_axis(length)
    noise = bandpass(rng.standard_normal(len(t)), 1500, 9000) * np.exp(-t / 0.07)
    body = (np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.045)
            + 0.5 * np.sin(2 * np.pi * 330 * t) * np.exp(-t / 0.03))
    snare = 0.8 * normalize(noise) + 0.6 * normalize(body)
    return fade_tail(normalize(np.tanh(2.0 * snare)))


def make_hat(length=0.06, decay=0.012):
    """Hi-hat cerrado: ruido blanco con pasa-altos y caída muy rápida."""
    t = t_axis(length)
    hat = highpass(rng.standard_normal(len(t)), 7000, order=4) * np.exp(-t / decay)
    return fade_tail(normalize(hat), 0.005)


# =============================================================================
# SECUENCIADOR
# =============================================================================
def render_cowbell(total, n_bars):
    """Melodía del cencerro en semicorcheas desde el compás 0 (intro incluida)."""
    buf = np.zeros(total)
    cache = {}
    for bar in range(n_bars):
        for s in range(STEPS_PER_BAR):
            idx = (bar * STEPS_PER_BAR + s) % len(COWBELL_MELODY)
            degree = COWBELL_MELODY[idx]
            if degree is None:
                continue
            semis = degree_to_semitones(degree)
            if semis not in cache:
                cache[semis] = make_cowbell(semis)
            accent = 1.0 if s % 4 == 0 else (0.85 if s % 2 == 0 else 0.7)
            add_at(buf, cache[semis], bar * BAR + s * STEP, accent)
    ir = make_reverb_ir(1.2, 0.3)
    return reverb(buf, ir, COWBELL_REVERB_WET)


def render_808(total, n_bars):
    """808 monofónico con glide exponencial entre notas y distorsión.

    Se construye una curva de frecuencia y otra de volumen para TODO el tema
    y se integra la fase, así el oscilador es continuo (sin clicks) y el
    glide se produce de forma natural de una nota a la siguiente.
    """
    freq = np.full(total, TONIC_HZ)
    amp = np.zeros(total)
    prev_f = None
    for bar in range(INTRO_BARS, n_bars):
        loop_bar = (bar - INTRO_BARS) % len(BASS_PROGRESSION)
        root = degree_to_semitones(BASS_PROGRESSION[loop_bar])
        for step, length, offset in BASS_PATTERN:
            start = bar * BAR + step * STEP
            i0 = int(start * SR)
            i1 = min(total, int((start + length * STEP) * SR))
            if i0 >= total:
                break
            n = i1 - i0
            t = np.arange(n) / SR
            f = TONIC_HZ * 2 ** ((root + offset) / 12)
            f_from = prev_f if prev_f is not None else f
            # Glide exponencial: de la nota anterior a la nueva en GLIDE_TIME seg.
            frac = np.clip(t / GLIDE_TIME, 0, 1)
            freq[i0:i1] = f_from * (f / f_from) ** frac
            # Envolvente: ataque 3 ms, caída exponencial, release 10 ms
            env = np.exp(-t / BASS_DECAY) * attack_ramp(n, 0.003)
            r = min(n, int(0.01 * SR))
            env[n - r:] *= np.linspace(1, 0, r)
            amp[i0:i1] = env
            prev_f = f
    phase = 2 * np.pi * np.cumsum(freq) / SR
    sub = np.sin(phase) * amp
    # Distorsión: tanh (armónicos impares) + un poco de rectificación (pares)
    dist = np.tanh(BASS_DRIVE * sub) + 0.3 * np.tanh(BASS_DRIVE * np.abs(sub))
    dist = highpass(dist, 30)                 # quita el DC de la rectificación
    bass = 0.6 * sub + 0.5 * lowpass(dist, 2500)
    return bass


def render_drums(total, n_bars):
    kick_s, snare_s, hat_s = make_kick(), make_snare(), make_hat()
    kicks = np.zeros(total)
    snares = np.zeros(total)
    hats = np.zeros(total)
    for bar in range(INTRO_BARS, n_bars):
        t0 = bar * BAR
        loop_bar = (bar - INTRO_BARS) % 4
        for s in KICK_STEPS:
            add_at(kicks, kick_s, t0 + s * STEP)
        for s in SNARE_STEPS:
            add_at(snares, snare_s, t0 + s * STEP)
        # Redobles de hi-hat (con crescendo) y las semicorcheas que reemplazan
        skip = set()
        for start, hits, span in HAT_ROLLS.get(loop_bar, []):
            skip.update(range(start, start + span))
            for h in range(hits):
                add_at(hats, hat_s, t0 + (start + h * span / hits) * STEP,
                       0.45 + 0.55 * h / max(1, hits - 1))
        for s in range(STEPS_PER_BAR):
            if s not in skip:
                add_at(hats, hat_s, t0 + s * STEP, 1.0 if s % 2 == 0 else 0.6)
    snares = reverb(snares, make_reverb_ir(1.8, 0.5), SNARE_REVERB_WET)
    return kicks, snares, hats


# =============================================================================
# MAIN
# =============================================================================
def main():
    total = int(DURATION * SR)
    n_bars = math.ceil(DURATION / BAR)

    cowbell = render_cowbell(total, n_bars)
    bass = render_808(total, n_bars)
    kick, snare, hats = render_drums(total, n_bars)

    mix = (VOLUMES["cowbell"] * cowbell
           + VOLUMES["kick"] * kick
           + VOLUMES["bass"] * bass
           + VOLUMES["snare"] * snare
           + VOLUMES["hats"] * hats)

    # --- Master: limpieza de subgraves, saturación suave, fade y normalización
    mix = highpass(mix, 25)
    mix = normalize(mix)
    mix = np.tanh(MASTER_DRIVE * mix) / np.tanh(MASTER_DRIVE)
    f = min(total, int(FADE_OUT * SR))
    mix[total - f:] *= np.linspace(1, 0, f)
    mix = normalize(mix, 10 ** (PEAK_DB / 20))

    wavfile.write(OUTPUT_FILE, SR, (mix * 32767).astype(np.int16))
    print(f"Listo: {OUTPUT_FILE}  |  {BPM} BPM  |  {DURATION:.1f} s  |  "
          f"{n_bars} compases ({INTRO_BARS} de intro)")


if __name__ == "__main__":
    main()
