"""Paket training portabel; nama kelas aman dan metadata tetap dapat ditelusuri."""
import csv
import json
import re
from io import BytesIO, StringIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from matplotlib.figure import Figure

from audio_analysis import decode_audio, spectrogram


def build_training_zip(rows, read_audio):
    if not rows:
        raise ValueError("Dataset kosong.")
    labels = sorted({row["label"] for row in rows})
    classes = {label: f"class_{index:03d}_" + (re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_")[:60] or "ayam")
               for index, label in enumerate(labels)}
    metadata = StringIO()
    writer = csv.writer(metadata)
    writer.writerow(["sample_id", "label", "class_folder", "audio_path", "image_path",
                     "duration_seconds", "created_at", "notes", "source_recording_id"])
    archive = BytesIO()
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        for row in rows:
            audio = read_audio(row["id"])
            samples = decode_audio(audio)
            frequencies, times, power = spectrogram(samples)
            db = 10 * np.log10(np.maximum(power, 1e-30) / max(float(power.max()), 1e-30))
            # Gambar training konsisten tanpa sumbu, teks atau label yang membocorkan kelas.
            fig = Figure(figsize=(2.24, 2.24), dpi=100)
            axis = fig.add_axes([0, 0, 1, 1])
            axis.imshow(db, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0,
                        extent=(0, len(samples) / 16000, 0, 8000))
            axis.set_axis_off()
            png = BytesIO()
            fig.savefig(png, format="png", dpi=100)
            folder = classes[row["label"]]
            audio_path = f"audio/{folder}/sample_{row['id']}.wav"
            image_path = f"images/{folder}/sample_{row['id']}.png"
            bundle.writestr(audio_path, audio)
            bundle.writestr(image_path, png.getvalue())
            safe = lambda text: "'" + text if text.startswith(("=", "+", "-", "@", "\t", "\r", "\n")) else text
            writer.writerow([row["id"], safe(row["label"]), folder, audio_path, image_path,
                             len(samples) / 16000, row["created_at"], safe(row["notes"]), row["id"]])
        bundle.writestr("metadata.csv", metadata.getvalue())
        bundle.writestr("labels.json", json.dumps({folder: label for label, folder in classes.items()}, ensure_ascii=False, indent=2))
        bundle.writestr("features.json", json.dumps(rows, ensure_ascii=False, indent=2))
        bundle.writestr("README.txt", EXPORT_GUIDE)
    return archive.getvalue()


EXPORT_GUIDE = """DATASET AYAM KETAWA

Isi:
- audio/<class_folder>/*.wav: audio mono 16 kHz ternormalisasi.
- images/<class_folder>/*.png: spektrogram 224x224 tanpa sumbu/teks.
- labels.json: pemetaan folder ke label asli (sumber label yang persis).
- metadata.csv: jalur relatif, metadata dan source_recording_id.
  Teks berawalan karakter formula diberi apostrof untuk keamanan spreadsheet.
- features.json: fitur MFCC baseline, metadata dan label asli.

KAGGLE:
Buat dataset baru, unggah ZIP lalu gunakan audio/ atau images/ dalam notebook.
Baca metadata.csv untuk jalur relatif dan labels.json untuk label asli.
Contoh tanpa library tambahan:
  import csv, json
  from pathlib import Path
  root = Path('/kaggle/input/NAMA-DATASET')
  rows = list(csv.DictReader((root / 'metadata.csv').open()))
  labels = json.loads((root / 'labels.json').read_text())
Sesuaikan root dengan lokasi hasil ekstraksi Kaggle. Model training belum disertakan.

TEACHABLE MACHINE:
Gunakan proyek IMAGE untuk images/. Ekstrak ZIP, buat satu kelas per label,
lalu unggah PNG dalam folder kelas tersebut. labels.json memuat label asli.
Ini melatih pengenal GAMBAR spektrogram, bukan pengenal audio mikrofon langsung.
Inferensi harus menghasilkan gambar dengan praproses yang sama:
mono16kHz, normalisasi, STFT FFT512/overlap352/windowTukey, daya dB relatif
puncak, -80..0dB, magma, sumbu frekuensi linear 0..8kHz, PNG224x224 tanpa teks.
Tidak dijamin ZIP/WAV dapat diimpor langsung ke proyek AUDIO Teachable Machine.
Jangan masukkan WAV ke proyek Image.

EVALUASI:
Tidak ada split otomatis: sesi rekaman belum memiliki ID terstruktur.
Pisahkan train/validation/test menurut sesi rekaman, bukan gambar/potongan acak.
Audio dan gambar dari source_recording_id yang sama wajib berada pada split sama.
Simpan sesi berbeda dalam catatan; jangan menilai model dengan rekaman latihnya.
Kumpulkan data seimbang per ayam, sesi/mikrofon berbeda, dan contoh noise/ayam lain.
Arsip memuat audio dan catatan lokal; periksa sebelum mempublikasikan ke Kaggle.
"""
