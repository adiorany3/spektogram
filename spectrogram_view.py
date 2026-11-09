"""Tampilan spektrogram lanjutan; pengaturan tidak mengubah fitur dataset."""
import csv
from io import BytesIO, StringIO

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from audio_analysis import SAMPLE_RATE, spectral_statistics, spectrogram, wav_bytes


def show_analysis(samples):
    st.subheader("Spektrogram lanjutan")
    duration = len(samples) / SAMPLE_RATE
    with st.expander("Pengaturan analisis", expanded=True):
        a, b, c = st.columns(3)
        st.caption("Arahkan penunjuk atau klik ikon (?) pada setiap pengaturan untuk bantuan. Mulai dengan FFT 1024, overlap 75%, dan window Hann; sesuaikan dengan pola suara yang ingin diamati.")
        n_fft = a.selectbox("Ukuran FFT", [256, 512, 1024, 2048, 4096], index=2,
                           help="Jumlah sampel per bingkai. FFT besar memisahkan frekuensi berdekatan dengan lebih baik, tetapi mengaburkan kejadian singkat. FFT kecil lebih baik untuk perubahan cepat. Pada 16 kHz, FFT 1024 mencakup 64 ms dengan jarak bin 15,625 Hz.")
        overlap_pct = b.slider("Overlap (%)", 0, 90, 75, step=5,
                              help="Persentase tumpang tindih antarbingkai. Nilai tinggi membuat langkah waktu lebih rapat dan tampilan lebih halus, tetapi menambah komputasi; tidak meningkatkan resolusi frekuensi atau menambah informasi rekaman.")
        window = c.selectbox("Window", ["hann", "hamming", "blackman", "tukey"],
                             help="Pembobotan tepi bingkai untuk mengurangi kebocoran spektral. Hann: pilihan umum; Hamming: alternatif penekanan sidelobe; Blackman: sidelobe lebih rendah dengan puncak lebih lebar; Tukey: kompromi tepi bertaper dan bagian tengah datar.")
        a, b, c = st.columns(3)
        scale = a.selectbox("Skala frekuensi", ["Linear", "Logaritmik", "Mel"],
                            help="Linear: jarak sama untuk selisih Hz sama. Logaritmik: memperjelas frekuensi rendah dan rasio harmonik; 0 Hz tidak ditampilkan. Mel: skala mendekati persepsi pendengaran. Ini hanya transformasi sumbu STFT, bukan filterbank Mel.")
        dynamic_range = b.slider("Rentang dinamis (dB)", 20, 120, 80, step=5,
                                help="Batas warna dari −nilai ini hingga 0 dB relatif puncak pada waktu pilihan. Nilai besar menampilkan komponen lemah dan kebisingan; nilai kecil menonjolkan komponen kuat. Bukan volume dB SPL terkalibrasi.")
        palette = c.selectbox("Palet warna", ["magma", "inferno", "viridis", "cividis", "gray"],
                              help="Mengubah pemetaan daya menjadi warna, bukan data. Magma/Inferno menonjolkan kontras, Viridis/Cividis membantu keterbacaan, Gray untuk tampilan abu-abu. Baca nilainya melalui bilah warna.")
        frequency_range = st.slider("Rentang frekuensi (Hz)", 0, SAMPLE_RATE // 2, (0, SAMPLE_RATE // 2), step=50,
                                    help="Zoom vertikal pada grafik, bukan filter audio. Statistik tetap memakai seluruh pita 0–8 kHz. Batas 8 kHz berasal dari audio analisis 16 kHz; rentang terlalu sempit akan ditolak.")
        time_range = st.slider("Rentang waktu (detik)", 0.0, float(duration), (0.0, float(duration)), step=0.01,
                              help="Memilih bagian rekaman untuk grafik, statistik dan unduhan WAV pilihan. Rentang harus cukup panjang untuk satu bingkai FFT. Identifikasi dan penyimpanan dataset tetap memakai seluruh audio.")
        overlay = st.checkbox("Tampilkan jejak dominan dan centroid", value=True,
                              help="Garis sian menunjukkan frekuensi berdaya paling besar per bingkai; garis hijau putus-putus menunjukkan pusat frekuensi berbobot daya (centroid). Dominan bukan nada dasar/F0 dan dapat mengikuti noise atau harmonik; nonaktifkan agar pola spektrogram tidak tertutup.")
        st.caption("Pengaturan ini berlaku untuk eksplorasi dan unduhan grafik/statistik. Fitur identifikasi dan gambar ekspor dataset training memakai pengaturan tetap, sehingga tetap konsisten.")
    start, end = (int(value * SAMPLE_RATE) for value in time_range)
    segment = samples[start:end]
    if len(segment) < n_fft:
        st.warning(f"Rentang waktu terlalu pendek untuk FFT {n_fft}. Perlebar rentang atau kecilkan FFT.")
        return
    low, high = frequency_range
    if (high <= low or high - low < SAMPLE_RATE / n_fft
            or (scale == "Logaritmik" and high <= max(low, SAMPLE_RATE / n_fft))):
        st.warning("Perlebar rentang frekuensi agar mencakup setidaknya satu bin FFT.")
        return
    overlap = int(n_fft * overlap_pct / 100)
    frequencies, times, power = spectrogram(segment, n_fft, overlap, window)
    times = times + start / SAMPLE_RATE
    statistics = spectral_statistics(frequencies, times, power)
    db = 10 * np.log10(np.maximum(power, 1e-30) / max(float(power.max()), 1e-30))
    dominant = frequencies[np.argmax(power.mean(axis=1))]
    a, b, c, d = st.columns(4)
    a.metric("Durasi pilihan", f"{len(segment) / SAMPLE_RATE:.3f} s")
    b.metric("Frekuensi dominan", f"{dominant:.1f} Hz")
    c.metric("Centroid rerata", f"{np.nanmean(statistics['centroid_hz']):.1f} Hz" if np.isfinite(statistics['centroid_hz']).any() else "Tidak ada sinyal")
    d.metric("RMS pilihan", f"{np.sqrt(np.mean(segment ** 2)):.4f}")
    st.caption(f"Resolusi bin: {SAMPLE_RATE / n_fft:.2f} Hz · window: {1000 * n_fft / SAMPLE_RATE:.1f} ms · hop: {1000 * (n_fft - overlap) / SAMPLE_RATE:.1f} ms. FFT besar memperinci frekuensi tetapi mengaburkan perubahan waktu.")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1, 4, 1]}, facecolor="#f8fafc")
    fig.suptitle("Analisa Suara (Spektrogram)", fontsize=16, fontweight="bold", color="#0f172a")
    for axis in axes:
        axis.set_facecolor("#ffffff")
        axis.tick_params(labelsize=9, colors="#334155")
        axis.spines[["top", "right"]].set_visible(False)
        axis.xaxis.label.set_color("#334155")
        axis.yaxis.label.set_color("#334155")
    for axis in (axes[0], axes[2]):
        axis.grid(alpha=0.2, linewidth=0.5)
        axis.set_axisbelow(True)
        axis.margins(y=0.15)
    stride = max(1, len(segment) // 20000)
    axes[0].plot(start / SAMPLE_RATE + np.arange(len(segment))[::stride] / SAMPLE_RATE,
                 segment[::stride], linewidth=0.65, color="#2563eb")
    axes[0].set(ylabel="Amplitudo", title="Gelombang suara ternormalisasi",
                xlim=(start / SAMPLE_RATE, end / SAMPLE_RATE))
    image = axes[1].pcolormesh(times, frequencies, db, shading="auto", cmap=palette,
                               vmin=-dynamic_range, vmax=0, rasterized=True)
    if scale == "Logaritmik":
        axes[1].set_yscale("log")
        low = max(low, SAMPLE_RATE / n_fft)
    elif scale == "Mel":
        axes[1].set_yscale("function", functions=(lambda hz: 2595 * np.log10(1 + hz / 700),
                                                  lambda mel: 700 * (10 ** (mel / 2595) - 1)))
    axes[1].set(ylim=(low, high), xlim=(start / SAMPLE_RATE, end / SAMPLE_RATE),
                ylabel=f"Frekuensi (Hz; {scale})", title="Spektrogram STFT — daya relatif puncak pilihan")
    if overlay:
        axes[1].plot(times, statistics["dominan_hz"], color="#22d3ee", linewidth=0.65, alpha=0.8, label="Dominan (bukan F0)")
        axes[1].plot(times, statistics["centroid_hz"], color="#a3e635", linewidth=0.8, linestyle="--", alpha=0.85, label="Centroid")
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
    fig.colorbar(image, ax=axes, label="Daya relatif (dB)", fraction=0.025, pad=0.02)
    axes[2].fill_between(times, statistics["rms"], color="#f59e0b", alpha=0.2)
    axes[2].plot(times, statistics["rms"], color="#b45309", linewidth=1)
    axes[2].set(xlabel="Waktu rekaman (detik)", ylabel="RMS bingkai",
                xlim=(start / SAMPLE_RATE, end / SAMPLE_RATE))
    st.pyplot(fig)
    png = BytesIO()
    fig.savefig(png, format="png", dpi=160)
    plt.close(fig)
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(statistics.keys())
    writer.writerows(zip(*statistics.values()))
    a, b, c = st.columns(3)
    a.download_button("Unduh spektrogram PNG", png.getvalue(), "spektrogram.png", "image/png")
    b.download_button("Unduh statistik CSV", output.getvalue(), "statistik-spektral.csv", "text/csv")
    c.download_button("Unduh audio pilihan WAV", wav_bytes(segment), "audio-pilihan.wav", "audio/wav")
    with st.expander("Statistik per bingkai & penjelasan"):
        st.dataframe(statistics, hide_index=True, width="stretch")
        st.caption("Statistik dihitung dari seluruh pita 0–8 kHz pada rentang waktu pilihan. Rentang frekuensi hanya membatasi tampilan. Mel adalah transformasi sumbu STFT, bukan filterbank Mel. Dominan bukan estimasi nada dasar (F0). Roll-off 85% adalah frekuensi yang mencakup 85% daya. dB dan RMS relatif audio ternormalisasi, bukan dB SPL. Bingkai hening ditandai NaN untuk statistik frekuensi. Zoom hanya untuk eksplorasi; penyimpanan dan identifikasi tetap memakai seluruh audio.")
