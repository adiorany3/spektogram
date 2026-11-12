"""Pemeriksaan mandiri: python test_app.py (tanpa dataset produksi)."""
import sqlite3
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from audio_analysis import (decode_audio, extract_features, rank_matches, wav_bytes,
                            spectrogram, spectral_statistics)
from storage import delete_sample, get_audio, list_samples, save_sample


def run():
    time = np.arange(32000) / 16000
    first = 0.5 * np.sin(2 * np.pi * 650 * time)
    other = 0.5 * np.sin(2 * np.pi * 2200 * time)
    audio = wav_bytes(first)
    samples = decode_audio(audio)
    features = extract_features(samples)
    assert features.shape == (39,) and np.isfinite(features).all()
    for count in (4800, 16000 * 121, 16000 * 180):  # Minimum, >2 menit, dan maksimum.
        tone = 0.5 * np.sin(2 * np.pi * 650 * np.arange(count) / 16000)
        assert len(decode_audio(wav_bytes(tone))) == count
    for count in (4799, 16000 * 180 + 1):
        tone = 0.5 * np.sin(2 * np.pi * 650 * np.arange(count) / 16000)
        try:
            decode_audio(wav_bytes(tone))
        except ValueError as exc:
            assert "Durasi audio terbaca" in str(exc)
        else:
            raise AssertionError("Durasi di luar batas diterima")
    # Simulasikan header yang melaporkan durasi salah, isi audio tetap 2 detik.
    import soundfile as sf
    with sf.SoundFile(BytesIO(audio)) as actual:
        with patch("audio_analysis.sf.SoundFile") as reader:
            source = reader.return_value.__enter__.return_value
            source.samplerate, source.channels = 16000, 1
            source.__len__.return_value = 16000 * 999
            source.read.side_effect = actual.read
            assert len(decode_audio(audio)) == 32000
            source.read.assert_called_once_with(16000 * 180 + 1, dtype="float32", always_2d=True)
    for invalid in (b"", b"not audio", wav_bytes(np.zeros(16000)), wav_bytes(first[:100])):
        try:
            decode_audio(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Audio tidak valid diterima")
    for n_fft in (256, 512, 1024, 2048, 4096):
        frequencies, times, power = spectrogram(samples, n_fft, n_fft // 2, "hann")
        stats = spectral_statistics(frequencies, times, power)
        assert power.shape == (n_fft // 2 + 1, len(times))
        assert abs(np.mean(stats["dominan_hz"]) - 650) <= 16000 / n_fft
        assert abs(np.mean(stats["centroid_hz"]) - 650) < 10
        assert np.allclose(stats["rms"], np.sqrt(np.mean(samples ** 2)), atol=0.02)
    for arguments in ((123, 10, "hann"), (512, 512, "hann"), (512, 100, "invalid")):
        try:
            spectrogram(samples, *arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("Parameter STFT tidak valid diterima")
    frequencies, times, power = spectrogram(np.zeros(16000))
    silent_stats = spectral_statistics(frequencies, times, power)
    assert np.isnan(silent_stats["dominan_hz"]).all()
    assert (silent_stats["rms"] == 0).all()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "test.sqlite3"
        save_sample("Ayam-A", "sesi 1", audio, features, 2.0, path)
        save_sample("Ayam-B", "sesi 2", wav_bytes(other), extract_features(other), 2.0, path)
        rows = list_samples(path)
        ranking = rank_matches(features, rows)
        assert ranking[0]["label"] == "Ayam-A" and ranking[0]["similarity"] == 100
        assert ranking[1]["similarity"] < ranking[0]["similarity"]
        assert get_audio(ranking[0]["sample_id"], path) == audio
        try:
            save_sample("Duplikat", "", audio, features, 2.0, path)
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("Duplikat diterima")
        try:
            save_sample(" ", "", audio, features, 2.0, path)
        except ValueError:
            pass
        else:
            raise AssertionError("Label kosong diterima")
        app_path = str(Path(__file__).with_name("app.py"))
        with patch("storage.list_samples", return_value=[]):
            app = AppTest.from_file(app_path).run(timeout=30)
            assert not app.exception, app.exception
            assert app.radio[0].options == ["Rekam mikrofon", "Unggah file"]
            assert app.radio[0].value == "Rekam mikrofon"
            app.radio[0].set_value("Unggah file").run()
            assert not app.exception, app.exception
        with patch("storage.list_samples", return_value=rows), patch("storage.get_audio", return_value=audio):
            app = AppTest.from_file(app_path).run(timeout=30)
            assert not app.exception, app.exception
            for button_label in ("Siapkan paket training Kaggle / Teachable Machine", "Siapkan ekspor dataset ZIP"):
                next(button for button in app.button if button.label == button_label).click().run(timeout=30)
                assert not app.exception, app.exception
        delete_sample(ranking[0]["sample_id"], path)
        assert len(list_samples(path)) == 1
    script = """import numpy as np
from spectrogram_view import show_analysis
show_analysis(0.5 * np.sin(2 * np.pi * 650 * np.arange(16000) / 16000))
"""
    view = AppTest.from_string(script).run(timeout=30)
    assert not view.exception, view.exception
    assert len(view.get("download_button")) == 4
    assert any(button.label == "Unduh PNG 3D" for button in view.get("download_button"))
    assert any(header.value == "Spektrogram 3D" for header in view.subheader)
    controls = list(view.selectbox) + list(view.slider) + list(view.checkbox)
    assert len(controls) == 9
    assert all(control.proto.help.strip() for control in controls), "Bantuan pengaturan belum lengkap"
    for scale in ("Logaritmik", "Mel", "Linear"):
        view.selectbox[2].set_value(scale).run(timeout=30)
        assert not view.exception, view.exception
    view.selectbox[0].set_value(4096).run(timeout=30)
    assert not view.exception, view.exception
    view.slider[3].set_value((0.0, 0.01)).run()
    assert not view.exception and len(view.warning) == 1
    import csv
    import json
    from zipfile import ZipFile
    from PIL import Image
    from dataset_export import build_training_zip
    export_rows = [{"id": i, "label": label, "notes": "=unsafe", "created_at": "test",
                    "features": features.tolist()} for i, label in enumerate(["../Ayam A", "../Ayam A", "Ayam A", "=Ayam"], 1)]
    with ZipFile(BytesIO(build_training_zip(export_rows, lambda _: audio))) as bundle:
        names = bundle.namelist()
        assert not any(".." in name or name.startswith("/") for name in names)
        mapping = json.loads(bundle.read("labels.json"))
        assert len(mapping) == 3
        metadata = list(csv.DictReader(StringIO(bundle.read("metadata.csv").decode())))
        assert len(metadata) == 4
        for row in metadata:
            assert row["notes"] == "'=unsafe"
            assert row["audio_path"] in names and row["image_path"] in names
            assert Image.open(BytesIO(bundle.read(row["image_path"]))).size == (224, 224)
            assert len(decode_audio(bundle.read(row["audio_path"]))) == 32000
        assert metadata[0]["class_folder"] == metadata[1]["class_folder"]
    print("LULUS: audio, MFCC, SQLite, UI, ekspor training, STFT, statistik, skala dan zoom.")


if __name__ == "__main__":
    run()
