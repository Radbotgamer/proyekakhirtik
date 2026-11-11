# ====================================================================
# APLIKASI WEB STREAMLIT: KATALOG EKSTRAKULER & CHATBOT AI
# SMA Swasta Kristen Kalam Kudus Pematangsiantar
# ====================================================================

import string
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import streamlit as st

# Konfigurasi Halaman Web Streamlit
st.set_page_config(
    page_title="Katalog Ekstrakurikuler Kalam Kudus 2026",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Katalog Ekstrakurikuler & Chatbot AI")
st.markdown(
    "**SMA Swasta Kristen Kalam Kudus Pematangsiantar (Tahun Ajaran"
    " 2026/2027)**"
)


# Fungsi Membaca Data dari Excel dengan Cache agar lebih cepat & stabil
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
  st.error(
      f"Gagal memuat file Excel. Pastikan file Excel berada di folder yang"
      f" sama! Error: {e}"
  )
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


# Pelabelan Kategori Otomatis untuk Naive Bayes
def tentukan_kategori(nama):
  nama_str = str(nama).lower()
  if (
      "osn" in nama_str
      or "olimpiade" in nama_str
      or "club" in nama_str
      or "social" in nama_str
      or "civic" in nama_str
  ):
    return "Akademik"
  elif "basket" in nama_str or "futsal" in nama_str or "badminton" in nama_str:
    return "Olahraga"
  elif "musik" in nama_str or "tari" in nama_str or "puisi" in nama_str:
    return "Seni & Musik"
  elif (
      "bahasa" in nama_str
      or "speaking" in nama_str
      or "orator" in nama_str
      or "book" in nama_str
  ):
    return "Bahasa"
  elif "programming" in nama_str or "stem" in nama_str or "design" in nama_str:
    return "Teknologi"
  elif "paskibra" in nama_str or "brigade" in nama_str:
    return "Keterampilan"
  elif "bible" in nama_str:
    return "Kerohanian"
  else:
    return "Umum"


df["kategori"] = df[kolom_ekskul].apply(tentukan_kategori)

df["deskripsi"] = (
    df[kolom_ekskul].astype(str)
    + " hari "
    + df[kolom_hari].astype(str)
    + " pukul "
    + df[kolom_waktu].astype(str)
    + " guru "
    + df[kolom_guru].astype(str)
    + " tempat "
    + df[kolom_tempat].astype(str)
    + " kuota "
    + df[kolom_kuota].astype(str)
)


# Preprocessing Teks & Pelatihan Model AI
def preprocessing_teks(teks):
  stopwords_indonesia = set([
      "dan",
      "atau",
      "di",
      "ke",
      "dari",
      "yang",
      "untuk",
      "pada",
      "adalah",
      "sebagai",
      "dengan",
      "ini",
      "itu",
      "dalam",
      "oleh",
      "karena",
  ])
  teks_kecil = str(teks).lower()
  teks_tanpa_tanda_baca = teks_kecil.translate(
      str.maketrans("", "", string.python if hasattr(string, 'python') else string.punctuation)
  )
  kata_kata = teks_tanpa_tanda_baca.split()
  kata_bersih = [
      kata for kata in kata_kata if kata not in stopwords_indonesia
  ]
  return " ".join(kata_bersih)


df["deskripsi_bersih"] = df["deskripsi"].apply(preprocessing_teks)

model_ai = make_pipeline(CountVectorizer(), MultinomialNB())
model_ai.fit(df["deskripsi_bersih"], df["kategori"])

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df["deskripsi"])

# --- TAMPILAN WEB (LAYOUT) ---
tab1, tab2, tab3 = st.tabs(
    ["📋 Tabel Katalog", "📊 Visualisasi Grafik", "🤖 Tanya Chatbot AI"]
)

with tab1:
  st.subheader("Daftar Kegiatan Ekstrakurikuler")
  st.dataframe(
      df[[kolom_hari, kolom_waktu, kolom_ekskul, kolom_guru, kolom_tempat, "kategori", kolom_kuota]],
      use_container_width=True,
  )

with tab2:
  st.subheader("Sebaran Jadwal Ekstrakurikuler Berdasarkan Hari")
  fig, ax = plt.subplots(figsize=(10, 4))
  urutan_hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"]
  sns.countplot(
      data=df,
      y=kolom_hari,
      order=[h for h in urutan_hari if h in df[kolom_hari].values],
      hue="kategori",
      palette="Set2",
      ax=ax,
  )
  ax.set_xlabel("Jumlah Kegiatan")
  ax.set_ylabel("Hari")
  st.pyplot(fig)

with tab3:
  st.subheader("Tanya Jawab Ekstrakurikuler dengan AI")
  st.write(
      "Contoh pertanyaan: *'Kapan jadwal ekskul basket?'*, *'Siapa pembimbing"
      " musik?'*, atau *'Kategori untuk OSN Biologi'*."
  )

  user_query = st.text_input("Ketik pertanyaan Anda di sini:")
  if user_query:
    q_bersih = preprocessing_teks(user_query)

    if "kategori" in q_bersih or "golongan" in q_bersih:
      teks_uji = q_bersih.replace("kategori", "").replace("golongan", "").strip()
      if len(teks_uji) > 2:
        pred = model_ai.predict([teks_uji])[0]
        st.success(
            f"**Hasil Analisis AI (Naive Bayes):** Masuk dalam kategori"
            f" **{pred}**."
        )
    else:
      q_vec = tfidf_vectorizer.transform([user_query])
      similarity = cosine_similarity(q_vec, tfidf_matrix)
      best_match_idx = similarity.argmax()
      best_score = similarity[0, best_match_idx]

      if best_score < 0.15:
        st.warning(
            "Maaf, informasi tidak ditemukan. Coba tanyakan nama ekskul atau"
            " hari pelaksanaan!"
        )
      else:
        row = df.iloc[best_match_idx]
        st.info(
            f"**Informasi Ditemukan (Kategori: {row['kategori']}):**\n\n"
            f"- **Nama Ekskul:** {row[kolom_ekskul]}\n"
            f"- **Hari / Waktu:** {row[kolom_hari]}, pukul {row[kolom_waktu]}\n"
            f"- **Pembimbing:** {row[kolom_guru]}\n"
            f"- **Tempat:** {row[kolom_tempat]}\n"
            f"- **Kuota:** {row[kolom_kuota]}"
        )