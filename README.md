# Spektrogram Ayam Ketawa

Aplikasi Streamlit lokal untuk analisis audio, pengumpulan dataset berlabel per ayam,
dan pencocokan rekaman baru dengan referensi tersimpan. Ini baseline MFCC, bukan
model identifikasi terlatih atau sistem dengan akurasi yang telah divalidasi.

## Menjalankan (macOS / Linux)

Gunakan Python 3.11–3.13. Lingkungan proyek ini telah disiapkan dengan Python 3.13.

```sh
cd /Users/macbookpro/Desktop/Spektogram
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Jika `.venv` sudah ada, cukup aktifkan dan jalankan perintah Streamlit.
Buka URL lokal yang ditampilkan, biasanya http://localhost:8501.

## Alur kerja

1. Unggah WAV/FLAC/OGG atau rekam melalui mikrofon browser.
2. Periksa gelombang, spektrogram, durasi, dan frekuensi dominan.
3. Simpan dengan label konsisten, misalnya `Ayam-001`; tambahkan catatan sesi.
4. Kumpulkan beberapa rekaman dari sesi berbeda untuk setiap ayam.
5. Uji rekaman baru yang belum menjadi referensi pada tab Analisis & identifikasi.
6. Kelola, putar, hapus, atau ekspor audio + metadata + fitur di tab Dataset.

Batas input: 20 MB, 0,3–60 detik, 8–192 kHz, maksimal 8 kanal.
Audio disimpan sebagai WAV mono 16 kHz yang dinormalisasi; simpan file asli terpisah
jika dibutuhkan untuk riset. Database: `data/samples.sqlite3`, dibuat otomatis.
Audio duplikat setelah normalisasi ditolak. Label dan audio disimpan lokal.
Untuk mencadangkan seluruh dataset, hentikan aplikasi lalu salin folder `data`.
Ekspor ZIP tersedia; impor belum diimplementasikan.

### Ekspor training Kaggle / Teachable Machine

Di tab Dataset, klik **Siapkan paket training Kaggle / Teachable Machine**, lalu unduh ZIP.
Paket berisi `audio/<kelas>/*.wav`, `images/<kelas>/*.png`, `metadata.csv`, `labels.json`,
`features.json`, dan panduan. Nama folder aman/unik; label asli ada di labels.json.
Gambar spektrogram konsisten 224×224 tanpa sumbu/teks, tidak mengikuti pengaturan visual.

Untuk Kaggle: unggah ZIP sebagai dataset lalu gunakan audio atau gambar di notebook.
Untuk Teachable Machine: ekstrak ZIP, buat proyek **Image**, buat kelas sesuai label,
lalu unggah PNG setiap kelas. Ini model gambar spektrogram, bukan audio mikrofon langsung.
Impor WAV/ZIP langsung ke proyek Audio tidak dijamin; gunakan alur yang didukung platform.
Pisahkan data berdasarkan sesi rekaman, bukan membagi acak gambar/audio dari rekaman sama.
Belum ada split otomatis atau script training; dataset ekspor tetap netral terhadap model.
Periksa catatan/audio sebelum dataset dipublikasikan.

## Metode dan batasan

### Spektrogram lanjutan

- FFT 256–4096 sampel, overlap 0–90%, window Hann/Hamming/Blackman/Tukey.
- Skala frekuensi linear, logaritmik, atau Mel (transformasi sumbu STFT, bukan filterbank).
- Zoom waktu/frekuensi, rentang dinamis 20–120 dB, serta pilihan palet warna.
- Jejak frekuensi dominan dan centroid, grafik RMS, statistik bandwidth dan roll-off 85%.
- Unduh grafik PNG, statistik per bingkai CSV, atau audio rentang pilihan WAV.

Statistik menggunakan seluruh pita 0–8 kHz pada waktu pilihan; zoom frekuensi hanya
memengaruhi tampilan. Dominan bukan F0; dB relatif puncak pilihan, bukan dB SPL.
FFT besar meningkatkan resolusi frekuensi dengan mengurangi resolusi waktu.
Audio tetap diolah pada 16 kHz sehingga frekuensi di atas 8 kHz tidak dianalisis.
Zoom tidak mengubah audio yang disimpan atau dipakai untuk identifikasi.

Ekstraksi fitur identifikasi tetap menggunakan jendela 512 sampel dan hop 160 sampel;
pengaturan visual tidak mengubah fitur referensi yang sudah tersimpan.
Fitur: statistik 13 MFCC non-energi dan perubahan antarbingkai (39 nilai).
Bingkai berenergi sangat rendah diabaikan. Kandidat dipilih dari sampel dengan
jarak fitur berskala tetap terkecil; skor `100 * exp(-jarak)` bukan probabilitas.
Ambang awal 75% harus dikalibrasi dengan data nyata; bukan jaminan suara dikenal.
Aplikasi tidak mendeteksi spesies dan belum memodelkan pola ritme ketawa secara utuh.

Pisahkan data latih/referensi dan uji berdasarkan sesi rekaman. Evaluasi juga suara
ayam lain, kebisingan, dan ayam yang tidak terdaftar sebelum penggunaan identifikasi.
Belum ada login; gunakan secara lokal, bukan layanan publik. Mikrofon memerlukan
izin browser dan localhost/HTTPS. Ekspor memuat audio: bagikan dengan hati-hati.

## Pemeriksaan mandiri

```sh
cd /Users/macbookpro/Desktop/Spektogram
.venv/bin/python test_app.py
```

Tes memakai sinyal sintetis dan database sementara: validasi audio, ekstraksi fitur,
ranking, operasi dataset, serta smoke test UI dan ekspor. Bukan pengukuran akurasi ayam.
