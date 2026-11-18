# ====================================================================
# KATALOG EKSTRAKULER & KLASIFIKASI NAIVE BAYES - SMA KALAM KUDUS
# ====================================================================

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="Katalog Ekstrakurikuler Kalam Kudus 2026",
    page_icon="🤖",
    layout="wide",
)

st.title("🎓 Katalog Ekstrakurikuler & Klasifikasi Cerdas (Naive Bayes)")
st.markdown(
    "**SMA Swasta Kristen Kalam Kudus Pematangsiantar (Tahun Ajaran"
    " 2026/2027)**"
)


# Fungsi Membaca Data Excel
@st.cache_data
def load_data():
  file_excel = "Jadwal_Ekstrakurikuler_Kalam_Kudus_2026.xlsx"
  df_raw = pd.read_excel(file_excel, header=None)

  header_row_index = None
  for idx, row in df_raw.iterrows():
    row_str = row.astype(str).str.cat(sep=" ").lower()
    if "hari" in row_str and "ekskul" in row_str:
      header_row_index = idx
      break

  if header_row_index is not None:
    df = pd.read_excel(file_excel, header=header_row_index)
  else:
    df = pd.read_excel(file_excel, header=2)

  df.columns = df.columns.astype(str).str.strip()
  return df


try:
  df = load_data()
except Exception as e:
  st.error(f"Gagal memuat file Excel. Pastikan file Excel ada di folder! {e}")
  st.stop()

# Deteksi Kolom Otomatis
kolom_hari = next(
    (c for c in df.columns if "hari" in c.lower()), df.columns[0]
)
kolom_ekskul = next(
    (c for c in df.columns if "ekskul" in c.lower() or "nama" in c.lower()),
    df.columns[2],
)
kolom_waktu = next(
    (c for c in df.columns if "waktu" in c.lower()), df.columns[1]
)
kolom_guru = next(
    (c for c in df.columns if "guru" in c.lower() or "pembimbing" in c.lower()),
    df.columns[3],
)
kolom_tempat = next(
    (c for c in df.columns if "tempat" in c.lower() or "kelas" in c.lower()),
    df.columns[4],
)
kolom_kuota = next(
    (c for c in df.columns if "kuota" in c.lower()), df.columns[5]
)

df = df.dropna(subset=[kolom_ekskul, kolom_hari])


# Fungsi Kategori Otomatis untuk Latihan Naive Bayes
def tentukan_kategori(nama):
  nama_str = str(nama).lower()
  if any(k in nama_str for k in ["basket", "futsal", "badminton", "voli"]):
    return "Olahraga"
  elif any(k in nama_str for k in ["musik", "tari", "choir", "vocal", "paduan"]):
    return "Seni & Musik"
  elif any(
      k in nama_str for k in ["osn", "olimpiade", "math", "fisika", "kimia"]
  ):
    return "Akademik"
  elif any(k in nama_str for k in ["english", "mandarin", "jepang"]):
    return "Bahasa"
  else:
    return "Keterampilan/Umum"


df["Kategori_Asli"] = df[kolom_ekskul].apply(tentukan_kategori)

# --- PELATIHAN MODEL NAIVE BAYES ---
# Menggunakan CountVectorizer & MultinomialNB untuk klasifikasi teks
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(df[kolom_ekskul].astype(str))
y_train = df["Kategori_Asli"]

model = MultinomialNB()
model.fit(X_train, y_train)

# Prediksi kembali data dengan model Naive Bayes
df["Prediksi_Kategori"] = model.predict(X_train)

# --- TAMPILAN WEB ---
st.subheader("📋 Daftar Ekstrakurikuler & Hasil Klasifikasi Naive Bayes")
st.dataframe(
    df[[kolom_hari, kolom_waktu, kolom_ekskul, "Prediksi_Kategori", kolom_guru, kolom_tempat, kolom_kuota]],
    use_container_width=True,
)

st.markdown("---")
st.subheader(
    "🤖 Sistem Tanya-Jawab / Prediksi Kategori dengan Naive Bayes"
)
st.write(
    "Ketik nama kegiatan atau kata kunci untuk diklasifikasikan oleh sistem"
    " Machine Learning:"
)

query_input = st.text_input(
    "Masukkan nama kegiatan (contoh: 'Klub Basket' atau 'Paduan Suara'):"
)

if query_input:
  # Vektorisasi dan prediksi input user menggunakan model Naive Bayes
  query_vec = vectorizer.transform([query_input])
  prediksi_hasil = model.predict(query_vec)[0]

  st.success(
      f"Hasil Analisis Naive Bayes: Kegiatan tersebut masuk ke kategori:"
      f" **{prediksi_hasil}**"
  )

  # Menampilkan data yang cocok dari tabel
  hasil_filter = df[df["Prediksi_Kategori"] == prediksi_hasil]
  if not hasil_filter.empty:
    st.write(
        "Rekomendasi kegiatan lain yang sejenis di kategori"
        f" **{prediksi_hasil}**:"
    )
    for idx, row in hasil_filter.iterrows():
      st.info(
          f"• **{row[kolom_ekskul]}** | Hari: {row[kolom_hari]} ({row[kolom_waktu]})"
          f" | Pembimbing: {row[kolom_guru]}"
      )

st.markdown("---")
st.subheader("🔍 Pencarian Bebas Berdasarkan Kata Kunci")
keyword = st.text_input("Cari bebas (hari, nama guru, dll):")
if keyword:
  hasil_cari = df[
      df.apply(
          lambda row: row.astype(str)
          .str.contains(keyword, case=False, na=False)
          .any(),
          axis=1,
      )
  ]
  if not hasil_cari.empty:
    st.write(f"Ditemukan {len(hasil_cari)} data:")
    st.dataframe(
        hasil_cari[[kolom_hari, kolom_waktu, kolom_ekskul, kolom_guru, kolom_tempat]]
    )
  else:
    st.warning("Data tidak ditemukan.")
