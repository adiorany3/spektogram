"""Validasi audio dan fitur baseline deterministik (versi 1)."""
from io import BytesIO
from math import gcd

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.fft import dct

SAMPLE_RATE = 16000
MAX_BYTES = 20 * 1024 * 1024
MAX_SECONDS = 180
FEATURE_VERSION = 1


def decode_audio(data):
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Audio harus berukuran 1 byte–20 MB.")
    try:
        with sf.SoundFile(BytesIO(data)) as source:
            if not 8000 <= source.samplerate <= 192000 or not 1 <= source.channels <= 8:
                raise ValueError("Gunakan audio 8–192 kHz dengan maksimal 8 kanal.")
            rate = source.samplerate
            # Header rekaman dapat melaporkan jumlah frame yang keliru.
            # Baca terbatas, lalu validasi jumlah sampel yang benar-benar terdekode.
            frames = source.read(MAX_SECONDS * rate + 1, dtype="float32", always_2d=True)
            duration = len(frames) / rate
            if not 0.3 <= duration <= MAX_SECONDS:
                measured = (f"lebih dari {MAX_SECONDS} detik" if duration > MAX_SECONDS
                            else f"{duration:.6f} detik")
                raise ValueError(
                    f"Durasi audio terbaca {measured}; batas yang didukung 0,3–{MAX_SECONDS} detik. "
                    "Jika durasi di pemutar berbeda, ekspor ulang sebagai WAV PCM lalu unggah kembali."
                )
            samples = frames.mean(axis=1)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise ValueError("Audio tidak terbaca. Gunakan WAV, FLAC, atau OGG yang valid.") from exc
    if not np.isfinite(samples).all():
        raise ValueError("Audio berisi nilai tidak valid.")
    samples -= samples.mean()
    if np.max(np.abs(samples)) < 1e-5:
        raise ValueError("Audio hening atau terlalu lemah untuk dianalisis.")
    divisor = gcd(rate, SAMPLE_RATE)
    samples = signal.resample_poly(samples, SAMPLE_RATE // divisor, rate // divisor)
    samples = samples / max(np.max(np.abs(samples)), 1e-8)
    return samples.astype(np.float32)


def spectrogram(samples, n_fft=512, overlap=None, window="tukey"):
    if n_fft not in (256, 512, 1024, 2048, 4096) or len(samples) < n_fft:
        raise ValueError("Ukuran FFT tidak didukung atau melebihi panjang audio.")
    overlap = min(352, n_fft - 1) if overlap is None else overlap
    if not isinstance(overlap, int) or not 0 <= overlap < n_fft:
        raise ValueError("Overlap harus berupa jumlah sampel antara 0 dan FFT−1.")
    if window not in ("tukey", "hann", "hamming", "blackman"):
        raise ValueError("Window tidak didukung.")
    frequencies, times, power = signal.spectrogram(
        samples, fs=SAMPLE_RATE, nperseg=n_fft, noverlap=overlap, window=window, mode="psd"
    )
    return frequencies, times, power


def spectral_statistics(frequencies, times, power):
    """Statistik PSD per bingkai; RMS setelah pengurangan DC oleh STFT."""
    total = power.sum(axis=0)
    weights = power / np.maximum(total, 1e-30)
    centroid = (frequencies[:, None] * weights).sum(axis=0)
    bandwidth = np.sqrt(((frequencies[:, None] - centroid) ** 2 * weights).sum(axis=0))
    rolloff = frequencies[np.argmax(np.cumsum(weights, axis=0) >= 0.85, axis=0)]
    peak = frequencies[np.argmax(power, axis=0)]
    rms = np.sqrt(total * (frequencies[1] - frequencies[0]))
    active = total > max(float(total.max()) * 1e-8, 1e-30)
    return {"waktu_detik": times, "dominan_hz": np.where(active, peak, np.nan),
            "centroid_hz": np.where(active, centroid, np.nan),
            "bandwidth_hz": np.where(active, bandwidth, np.nan),
            "rolloff85_hz": np.where(active, rolloff, np.nan), "rms": rms}


def extract_features(samples):
    _, _, power = spectrogram(samples)
    # Buang bingkai sangat lemah agar jeda panjang tidak mendominasi fitur.
    energy = power.sum(axis=0)
    power = power[:, energy > max(energy.max() * 0.01, 1e-12)]
    if power.shape[1] < 3:
        raise ValueError("Bagian suara aktif terlalu pendek; rekam ulang lebih dekat.")
    mel_edges = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 42) / 2595) - 1)
    frequencies = np.linspace(0, SAMPLE_RATE / 2, power.shape[0])
    filters = []
    for left, center, right in zip(mel_edges[:-2], mel_edges[1:-1], mel_edges[2:]):
        filters.append(np.maximum(0, np.minimum((frequencies - left) / (center - left),
                                                (right - frequencies) / (right - center))))
    log_mel = np.log(np.maximum(np.asarray(filters) @ power, 1e-12))
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[1:14]
    delta = np.diff(mfcc, axis=1)
    vector = np.concatenate((mfcc.mean(axis=1), mfcc.std(axis=1), np.abs(delta).mean(axis=1)))
    return vector


def rank_matches(query, rows):
    """Jarak berskala tetap: hasil stabil walau dataset hanya satu sampel."""
    # ponytail: baseline MFCC tanpa model terlatih; ganti setelah ada dataset evaluasi berlabel.
    scale = np.tile(np.array([20.0, 10.0, 10.0]), (13, 1)).T.reshape(-1)
    matches = []
    for row in rows:
        distance = float(np.sqrt(np.mean(((query - np.asarray(row["features"])) / scale) ** 2)))
        matches.append({"label": row["label"], "sample_id": row["id"],
                        "distance": distance, "similarity": 100 * np.exp(-distance)})
    return sorted(matches, key=lambda row: row["distance"])


def wav_bytes(samples):
    output = BytesIO()
    sf.write(output, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return output.getvalue()
