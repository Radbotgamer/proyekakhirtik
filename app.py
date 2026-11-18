# ====================================================================
# KATALOG EKSTRAKULER & PENCARIAN CERDAS - SMA KALAM KUDUS
# ====================================================================

import pandas as pd
import streamlit as st

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="Katalog Ekstrakurikuler Kalam Kudus 2026",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Katalog Ekstrakurikuler & Informasi Cerdas")
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

# Tampilan Utama Website
st.subheader("📋 Daftar Lengkap Kegiatan Ekstrakurikuler")
st.dataframe(
    df[[kolom_hari, kolom_waktu, kolom_ekskul, kolom_guru, kolom_tempat, kolom_kuota]],
    use_container_width=True,
)

st.markdown("---")
st.subheader("🔍 Cari Informasi Ekstrakurikuler")
keyword = st.text_input(
    "Ketik nama ekskul atau hari (contoh: Basket, Musik, Senin):"
)

if keyword:
  hasil = df[
      df.apply(
          lambda row: row.astype(str)
          .str.contains(keyword, case=False, na=False)
          .any(),
          axis=1,
      )
  ]

  if not hasil.empty:
    st.success(f"Ditemukan {len(hasil)} kegiatan yang cocok:")
    for idx, row in hasil.iterrows():
      st.info(
          f"• **{row[kolom_ekskul]}**\n"
          f"  - Hari / Waktu: {row[kolom_hari]}, pukul {row[kolom_waktu]}\n"
          f"  - Pembimbing: {row[kolom_guru]}\n"
          f"  - Tempat: {row[kolom_tempat]}\n"
          f"  - Kuota: {row[kolom_kuota]}"
      )
  else:
    st.warning("Maaf, data tidak ditemukan. Coba kata kunci lain!")
