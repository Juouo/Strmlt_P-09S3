"""
Lab 1 - Eksperimen Chunking (15 menit)
P-10 RAG Part 1: Embedding & Vector Database

Langkah di modul:
1. Baca satu PDF dengan baca_pdf() dan cetak 500 karakter pertama.
2. Potong dengan potong() memakai tiga ukuran (500, 1000, 1500) dan overlap 10-20%.
   Catat jumlah chunk tiap pengaturan.
3. Lihat 3 chunk acak untuk tiap ukuran: adakah yang terpotong di tengah
   kalimat atau tabel?

Cara pakai:
    pip install pypdf
    python lab1_chunking.py                  # pakai PDF pertama di folder dokumen/
    python lab1_chunking.py dokumen/P-10.pdf # atau sebutkan file-nya
"""
import random
import re
import sys
from pathlib import Path

from pypdf import PdfReader

# ---------------------------------------------------------------- pengaturan
FOLDER = Path("dokumen")
UKURAN_LIST = [500, 1000, 1500]   # tiga ukuran chunk yang diminta modul
PERSEN_TUMPANG = 0.15             # overlap 15% (rentang modul: 10-20%)
JUMLAH_SAMPEL = 3                 # 3 chunk acak per ukuran
random.seed(42)                   # supaya sampel acak bisa diulang

LAPORAN = []


def tulis(teks=""):
    """Cetak ke layar dan simpan untuk lab1_hasil.txt."""
    print(teks)
    LAPORAN.append(teks)


# --------------------------------------------------- fungsi dari modul (6.2)
def baca_pdf(path):
    reader = PdfReader(path)
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def potong(teks, ukuran=800, tumpang=100):
    """Pemotongan ukuran tetap dengan overlap (versi paling sederhana)."""
    hasil, mulai = [], 0
    while mulai < len(teks):
        hasil.append(teks[mulai:mulai + ukuran])
        mulai += ukuran - tumpang
    return [h for h in hasil if h.strip()]


# ------------------------------------------------------------ alat bantu cek
def berakhir_di_kalimat(chunk):
    return chunk.rstrip().endswith((".", "!", "?"))


def diawali_tengah_kalimat(chunk):
    """Chunk yang dimulai huruf kecil hampir pasti dipotong di tengah kalimat."""
    awal = chunk.lstrip()[:1]
    return awal.isalpha() and awal.islower()


def mirip_tabel_atau_daftar(chunk):
    """Heuristik kasar: ada baris dengan 3 angka atau lebih, atau baris bernomor/bullet."""
    for baris in chunk.splitlines():
        if len(re.findall(r"\d+", baris)) >= 3:
            return True
        if re.match(r"\s*(\d+[.)]|[-•*])\s+", baris):
            return True
    return False


def cari_pdf():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    daftar = sorted(FOLDER.glob("*.pdf"))
    if not daftar:
        sys.exit(f"Tidak ada PDF di folder '{FOLDER}/'. Letakkan 1 file PDF di sana "
                 f"atau jalankan: python lab1_chunking.py path/ke/file.pdf")
    return daftar[0]


# ------------------------------------------------------------------- langkah 1
pdf = cari_pdf()
teks = baca_pdf(pdf)

tulis(f"=== LANGKAH 1: Baca PDF ({pdf.name}) ===")
tulis(f"Total karakter terbaca: {len(teks)}")
if len(teks.strip()) == 0:
    sys.exit("Teks kosong. PDF ini kemungkinan hasil scan. "
             "Pakai PDF berbasis teks atau lakukan OCR terlebih dahulu.")
tulis("500 karakter pertama:")
tulis("-" * 60)
tulis(teks[:500])
tulis("-" * 60)
tulis("Periksa: apakah kalimat terbaca utuh, tanpa karakter aneh atau spasi berantakan?\n")

# ------------------------------------------------------------------- langkah 2
tulis("=== LANGKAH 2: Jumlah chunk per ukuran ===")
tulis(f"{'ukuran':>7} {'tumpang':>8} {'jumlah chunk':>13} {'rata2 panjang':>14} "
      f"{'tak akhir kalimat':>18} {'mulai huruf kecil':>18}")

hasil_potong = {}
for ukuran in UKURAN_LIST:
    tumpang = int(ukuran * PERSEN_TUMPANG)
    chunks = potong(teks, ukuran, tumpang)
    hasil_potong[ukuran] = (tumpang, chunks)

    rata = sum(len(c) for c in chunks) / len(chunks)
    n_tak_akhir = sum(not berakhir_di_kalimat(c) for c in chunks)
    n_awal_kecil = sum(diawali_tengah_kalimat(c) for c in chunks)
    tulis(f"{ukuran:>7} {tumpang:>8} {len(chunks):>13} {rata:>14.0f} "
          f"{n_tak_akhir:>5}/{len(chunks):<12} {n_awal_kecil:>5}/{len(chunks):<12}")
tulis()

# ------------------------------------------------------------------- langkah 3
tulis("=== LANGKAH 3: 3 chunk acak per ukuran ===")
ringkasan = []
for ukuran in UKURAN_LIST:
    tumpang, chunks = hasil_potong[ukuran]
    sampel_idx = sorted(random.sample(range(len(chunks)), min(JUMLAH_SAMPEL, len(chunks))))
    tulis(f"\n##### UKURAN {ukuran} (overlap {tumpang}) #####")

    n_terpotong, n_tabel = 0, 0
    for i in sampel_idx:
        c = chunks[i]
        potong_kalimat = (not berakhir_di_kalimat(c)) or diawali_tengah_kalimat(c)
        tabel = mirip_tabel_atau_daftar(c)
        n_terpotong += potong_kalimat
        n_tabel += tabel

        tulis(f"\n[chunk {i} dari {len(chunks)} | {len(c)} karakter]")
        tulis(c)
        tulis(f">> terpotong di tengah kalimat : {'YA' if potong_kalimat else 'tidak'}")
        tulis(f">> mirip tabel/daftar          : {'YA, cek manual' if tabel else 'tidak'}")
    ringkasan.append((ukuran, n_terpotong, n_tabel, len(sampel_idx)))

tulis("\n=== RINGKASAN SAMPEL (bahan catatan lab) ===")
tulis(f"{'ukuran':>7} {'terpotong kalimat':>19} {'mirip tabel/daftar':>20}")
for ukuran, nt, nb, n in ringkasan:
    tulis(f"{ukuran:>7} {nt:>14}/{n:<4} {nb:>15}/{n:<4}")
tulis("\nCatatan: penanda di atas adalah heuristik. Baca sendiri tiap chunk dan "
      "pastikan penilaiannya, terutama untuk tabel dan daftar.")

Path("lab1_hasil.txt").write_text("\n".join(LAPORAN), encoding="utf-8")
print("\nHasil disimpan ke lab1_hasil.txt")
