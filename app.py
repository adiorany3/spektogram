"""Jalankan: streamlit run app.py."""
import csv
import json
import sqlite3
from io import BytesIO, StringIO
from zipfile import ZIP_DEFLATED, ZipFile

import streamlit as st

from audio_analysis import (FEATURE_VERSION, MAX_SECONDS, SAMPLE_RATE, decode_audio,
                            extract_features, rank_matches, wav_bytes)
from spectrogram_view import show_analysis
from dataset_export import build_training_zip
from storage import delete_sample, get_audio, list_samples, save_sample

st.set_page_config(page_title="Spektrogram Ayam Ketawa", page_icon="🐓", layout="wide")
st.title("🐓 Analisa Suara (Spektrogram)")
st.caption("Spektrogram • Dataset berlabel • Pencocokan suara berbasis fitur MFCC")
st.info("Alat analisis spektrogram dan pencocokan suara berdasarkan dataset. Hasil menunjukkan kemiripan, bukan kepastian identitas sumber suara. Kebisingan latar, mikrofon, dan kondisi perekaman dapat memengaruhi hasil.")


@st.cache_data(max_entries=8, show_spinner=False)
def analyze(data):
    samples = decode_audio(data)
    return samples, extract_features(samples)


input_tab, dataset_tab, guide_tab = st.tabs(["Analisis & identifikasi", "Dataset", "Panduan"])
with input_tab:
    source = st.radio("Sumber audio", ["Rekam mikrofon", "Unggah file"], horizontal=True)
    uploaded = (st.file_uploader(f"Audio WAV, FLAC, atau OGG · maksimal 20 MB / {MAX_SECONDS} detik",
                               type=["wav", "flac", "ogg"])
                if source == "Unggah file" else st.audio_input(f"Rekam suara ayam (0,3–{MAX_SECONDS} detik)"))
    if uploaded is not None:
        try:
            samples, features = analyze(uploaded.getvalue())
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.audio(uploaded.getvalue())
            show_analysis(samples)
            st.subheader("Cocokkan dengan dataset")
            rows = list_samples()
            compatible = [row for row in rows if row["feature_version"] == FEATURE_VERSION]
            if len(compatible) != len(rows):
                st.warning("Sampel dengan versi fitur berbeda tidak dipakai.")
            if not compatible:
                st.warning("Dataset kosong. Simpan beberapa rekaman berlabel terlebih dahulu.")
            else:
                threshold = st.slider("Ambang kemiripan awal (%) — perlu dikalibrasi", 0, 100, 75)
                matches = rank_matches(features, compatible)
                best = matches[0]
                if best["similarity"] >= threshold:
                    st.success(f"Kandidat terdekat: {best['label']} — kemiripan {best['similarity']:.1f}%")
                else:
                    st.warning("Tidak ada kecocokan memadai; perlakukan sebagai suara belum dikenal.")
                st.caption("Skor bukan probabilitas. Rekaman yang sama dengan referensi menghasilkan skor sangat tinggi.")
                st.dataframe([{"Label": item["label"], "ID sampel": item["sample_id"],
                               "Kemiripan (%)": round(item["similarity"], 1),
                               "Jarak fitur": round(item["distance"], 3)} for item in matches[:10]],
                             hide_index=True, width="stretch")
            st.subheader("Tambahkan ke dataset")
            with st.form("save_sample", clear_on_submit=True):
                label = st.text_input("Label identitas ayam", max_chars=100, placeholder="Contoh: Ayam-001")
                notes = st.text_area("Catatan sesi / jenis suara / kondisi rekaman", max_chars=2000)
                submitted = st.form_submit_button("Simpan sampel")
            if submitted:
                try:
                    save_sample(label, notes, wav_bytes(samples), features, len(samples) / SAMPLE_RATE)
                except sqlite3.IntegrityError:
                    st.warning("Audio ini sudah tersimpan. Gunakan rekaman lain, bukan duplikat.")
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.success("Sampel disimpan. Pencocokan diperbarui pada interaksi berikutnya.")

with dataset_tab:
    rows = list_samples()
    st.metric("Jumlah sampel", len(rows))
    st.caption(f"{len(set(row['label'] for row in rows))} label identitas · database lokal: data/samples.sqlite3")
    if rows:
        st.dataframe([{key: value for key, value in row.items() if key != "features"} for row in rows],
                     hide_index=True, width="stretch")
        selected = st.selectbox("Pilih sampel untuk diputar atau dihapus", [row["id"] for row in rows],
                                format_func=lambda value: next(f"#{r['id']} — {r['label']}" for r in rows if r["id"] == value))
        st.audio(get_audio(selected), format="audio/wav")
        confirmed = st.checkbox("Saya yakin ingin menghapus sampel terpilih")
        if st.button("Hapus sampel", disabled=not confirmed):
            delete_sample(selected)
            st.rerun()
        st.subheader("Ekspor untuk training")
        st.caption("Paket training: WAV + spektrogram PNG 224×224 per kelas, metadata CSV, labels.json, fitur dan panduan. Kaggle dapat memakai audio/gambar. Teachable Machine: gunakan proyek Image untuk PNG, bukan impor WAV langsung ke proyek Audio.")
        st.info("Pisahkan data uji berdasarkan sesi rekaman. Periksa audio dan catatan sebelum memublikasikan dataset.")
        if st.button("Siapkan paket training Kaggle / Teachable Machine"):
            try:
                with st.spinner("Membuat audio, spektrogram dan metadata…"):
                    training_zip = build_training_zip(rows, get_audio)
            except (ValueError, OSError) as exc:
                st.error(f"Ekspor training gagal: {exc}")
            else:
                st.download_button("Unduh paket training ZIP", training_zip,
                                   "training-ayam-ketawa.zip", "application/zip")
        if st.button("Siapkan ekspor dataset ZIP"):
            archive = BytesIO()
            metadata = StringIO()
            writer = csv.writer(metadata)
            writer.writerow(["id", "label", "notes", "created_at", "duration", "feature_version", "audio_path"])
            with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
                for row in rows:
                    audio_path = f"audio/{row['id']}.wav"
                    # Hindari formula injection saat CSV dibuka sebagai spreadsheet.
                    safe = lambda text: "'" + text if text.startswith(("=", "+", "-", "@", "\t", "\r", "\n")) else text
                    writer.writerow([row["id"], safe(row["label"]), safe(row["notes"]), row["created_at"],
                                     row["duration"], row["feature_version"], audio_path])
                    bundle.writestr(audio_path, get_audio(row["id"]))
                bundle.writestr("metadata.csv", metadata.getvalue())
                bundle.writestr("features.json", json.dumps(rows, ensure_ascii=False, indent=2))
            st.download_button("Unduh ZIP (audio, metadata, fitur)", archive.getvalue(),
                               "dataset-ayam-ketawa.zip", "application/zip")
    else:
        st.info("Belum ada sampel tersimpan.")

with guide_tab:
    st.markdown("""
### Cara menggunakan
1. Rekam satu ayam dalam satu file, sedekat mungkin dan tanpa suara ayam lain.
2. Unggah audio atau gunakan mikrofon browser; periksa gelombang dan spektrogram.
3. Isi label konsisten (misalnya **Ayam-001**), simpan beberapa sesi berbeda per ayam.
4. Uji dengan **rekaman baru yang tidak tersimpan sebagai referensi**.
5. Ekspor dataset dari tab Dataset; cadangkan folder `data` saat aplikasi berhenti.

### Metode dan keterbatasan
Audio diubah ke mono 16 kHz dan amplitudonya dinormalisasi. Spektrogram memakai STFT;
pencocokan memakai rerata, simpangan baku, dan perubahan 13 koefisien MFCC.
Jarak terdekat antar-sampel menghasilkan kandidat label, bukan model klasifikasi terlatih.
Aplikasi tidak memastikan bahwa sumber audio benar-benar ayam ketawa.

Ambang 75% hanyalah nilai awal, bukan hasil validasi ilmiah. Kalibrasikan memakai
rekaman uji terpisah per sesi dan contoh ayam yang tidak ada dalam dataset.
Suara serupa dapat tertukar; fitur ringkasan belum memodelkan urutan ritme ketawa.
Tidak ada jaminan akurasi sebelum dataset nyata dievaluasi.

### Penyimpanan dan privasi
Audio tersimpan lokal sebagai WAV ternormalisasi, bukan salinan file asli.
Ekspor berisi audio, metadata CSV, dan fitur JSON; belum tersedia impor dataset.
Aplikasi belum memiliki login: gunakan secara lokal saja.
Mikrofon memerlukan izin browser dan localhost atau HTTPS.
""")

st.divider()
st.caption("Developed by Galuh Adi Insani")
