"""
Lab 3 - Ukur Hit@k (25 menit)
P-10 RAG Part 1: Embedding & Vector Database

Langkah di modul:
1. Buat 15 pertanyaan uji beserta kata kunci jawaban dari dokumen Anda.
2. Jalankan hit_at_k untuk k = 1, 3, 5 pada DUA koleksi dengan ukuran chunk berbeda.
3. Tulis kesimpulan 3-4 kalimat: ukuran mana yang lebih baik, pada pertanyaan apa
   gagal, dan dugaan penyebabnya.

Cara pakai:
    1) Isi UJI di bawah dengan 15 pasangan (pertanyaan, kata/frasa kunci).
    2) python lab3_hit_at_k.py

Kata kunci harus kata/frasa yang PASTI ada di chunk jawaban yang benar,
persis seperti tertulis di PDF (huruf besar/kecil tidak berpengaruh).
"""
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# ---------------------------------------------------------------- pengaturan
FOLDER = Path("dokumen")
KS = (1, 3, 5)

# (nama koleksi, ukuran chunk, overlap). Overlap 15% (rentang modul: 10-20%).
# Tambahkan (..., 1500, 225) bila ingin membandingkan tiga ukuran seperti Proyek 4.
KONFIG = [
    ("pedoman_500", 500, 75),
    ("pedoman_1000", 1000, 150),
]

# (pertanyaan, kata/frasa yang HARUS ada di chunk jawaban yang benar)
# Dua contoh pertama dari modul. Ganti semuanya dengan pasangan dari dokumen Anda
# sampai berjumlah 15. Kata kunci boleh berupa satu string atau daftar alternatif,
# misalnya ["75 persen", "75%"].
# (pertanyaan, kata/frasa yang HARUS ada di chunk jawaban yang benar)
UJI = [
    # 1. Panduan Akademik Mahasiswa
    ("Apa syarat masa studi sebelum mahasiswa dapat mengajukan cuti akademik?",
     "sekurang-kurangnya dua semester aktif"),

    ("Apa batas minimal kehadiran agar mahasiswa dapat mengikuti ujian akhir semester?",
     "minimal 75% kehadiran"),

    ("Apa yang digambarkan oleh Indeks Prestasi Semester atau IPS?",
     "satu semester"),

    ("Prestasi dari seluruh semester yang telah ditempuh disebut apa?",
     "seluruh semester yang telah ditempuh"),

    ("Apa kepanjangan KRS dalam sistem akademik mahasiswa?",
     "Kartu Rencana Studi (KRS)"),

    ("Apa yang perlu dipastikan mahasiswa sebelum melakukan pengisian KRS?",
     "status administrasi telah aktif"),

    ("Dokumen apa yang perlu diisi saat mengajukan cuti akademik?",
     "Formulir permohonan dan alasan tertulis"),

    # 2. Panduan Administrasi dan Etika Digital
    ("Apa kegunaan Kartu Hasil Studi atau KHS?",
     "Melihat nilai mata kuliah per"),

    ("Apa pengertian phishing menurut glosarium dokumen?",
     "informasi sensitif melalui pesan atau situs"),

    ("Apa yang tidak boleh dibagikan kepada orang lain untuk menjaga keamanan akun?",
     "kode OTP"),

    ("Bagaimana kata sandi yang disarankan untuk akun akademik?",
     "kata sandi unik"),

    ("Apa yang perlu diperiksa sebelum memasukkan informasi login pada sebuah situs?",
     "alamat situs"),

    ("Dokumen apa yang mendukung permohonan mahasiswa untuk magang?",
     "Surat pengantar magang"),

    ("Melalui saluran apa permohonan administrasi sebaiknya diajukan?",
     "kanal resmi"),

    ("Bagaimana parafrase harus ditulis dalam kegiatan akademik?",
     "susunan kalimat sendiri"),
]

LAPORAN = []


def tulis(teks=""):
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


# ------------------------------------------------------------ validasi masukan
if len(UJI) != 15:
    sys.exit(f"UJI berisi {len(UJI)} pasangan. Modul meminta 15.")
if any(q.startswith("GANTI") or str(k).startswith("GANTI") for q, k in UJI):
    sys.exit("Masih ada pasangan UJI yang berisi 'GANTI'. Isi semua dengan pertanyaan "
             "dan kata kunci dari dokumen Anda, lalu jalankan ulang.")

pdf_list = sorted(FOLDER.glob("*.pdf"))
if not pdf_list:
    sys.exit(f"Tidak ada PDF di folder '{FOLDER}/'.")

# ------------------------------------------------ langkah 2a: bangun dua koleksi
# Satu embedding_function yang sama untuk semua koleksi (aturan emas modul).
emb = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="./chroma_db")

teks_pdf = {pdf: baca_pdf(pdf) for pdf in pdf_list}
koleksi_map = {}

tulis("=== Membangun koleksi ===")
for nama, ukuran, tumpang in KONFIG:
    try:
        client.delete_collection(nama)      # mulai bersih agar tidak ada chunk lama
    except Exception:
        pass
    koleksi = client.get_or_create_collection(
        name=nama, embedding_function=emb,
        metadata={"hnsw:space": "cosine"})

    total = 0
    for pdf, teks in teks_pdf.items():
        chunks = potong(teks, ukuran, tumpang)
        if not chunks:
            tulis(f"  {nama}: {pdf.name} kosong (mungkin PDF scan), dilewati")
            continue
        koleksi.upsert(
            ids=[f"{pdf.stem}-{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"sumber": pdf.name, "chunk": i} for i in range(len(chunks))],
        )
        total += len(chunks)
    koleksi_map[nama] = koleksi
    tulis(f"  {nama}: ukuran {ukuran}, overlap {tumpang} -> {total} chunk")
tulis()


# ------------------------------------------------------------------- evaluasi
def daftar_kunci(kunci):
    return [kunci] if isinstance(kunci, str) else list(kunci)


def peringkat_benar(koleksi, pertanyaan, kunci, k_maks):
    """Peringkat (mulai dari 1) chunk pertama yang memuat kata kunci. None bila tidak ada."""
    docs = koleksi.query(query_texts=[pertanyaan], n_results=k_maks)["documents"][0]
    for r, d in enumerate(docs, 1):
        if any(kk.lower() in d.lower() for kk in daftar_kunci(kunci)):
            return r, docs
    return None, docs


def hit_at_k(koleksi, uji, k=3):
    """Versi modul: proporsi pertanyaan yang chunk benarnya ada di k hasil teratas."""
    benar = 0
    for pertanyaan, kunci in uji:
        docs = koleksi.query(query_texts=[pertanyaan], n_results=k)["documents"][0]
        if any(kk.lower() in d.lower() for kk in daftar_kunci(kunci) for d in docs):
            benar += 1
    return benar / len(uji)


k_maks = max(KS)
hasil = {}   # nama -> {"rank": [...], "top1": [...]}
for nama, _, _ in KONFIG:
    ranks, top1 = [], []
    for pertanyaan, kunci in UJI:
        r, docs = peringkat_benar(koleksi_map[nama], pertanyaan, kunci, k_maks)
        ranks.append(r)
        top1.append(docs[0] if docs else "")
    hasil[nama] = {"rank": ranks, "top1": top1}

# Pastikan fungsi hit_at_k versi modul memberi angka sama dengan hitungan peringkat.
tulis("=== HASIL: Hit@k (dan MRR dari modul bagian 7) ===")
tulis(f"{'koleksi':<16} {'ukuran':>7} " + " ".join(f"{'Hit@'+str(k):>7}" for k in KS) + f" {'MRR@'+str(k_maks):>8}")
for nama, ukuran, _ in KONFIG:
    ranks = hasil[nama]["rank"]
    cells = []
    for k in KS:
        dari_peringkat = sum(1 for r in ranks if r is not None and r <= k) / len(UJI)
        dari_fungsi = hit_at_k(koleksi_map[nama], UJI, k)
        assert abs(dari_peringkat - dari_fungsi) < 1e-9, "Hit@k tidak konsisten"
        cells.append(f"{dari_fungsi:>7.2f}")
    mrr = sum((1 / r) if r else 0 for r in ranks) / len(UJI)
    tulis(f"{nama:<16} {ukuran:>7} " + " ".join(cells) + f" {mrr:>8.3f}")
tulis()

# ------------------------------------------------------ pertanyaan yang gagal
tulis(f"=== Pertanyaan yang gagal masuk top-{k_maks} (bahan kesimpulan) ===")
for nama, _, _ in KONFIG:
    tulis(f"\n[{nama}]")
    gagal = [(i, q, kunci) for i, (q, kunci) in enumerate(UJI) if hasil[nama]["rank"][i] is None]
    if not gagal:
        tulis("  Tidak ada. Semua pertanyaan terambil.")
    for i, q, kunci in gagal:
        tulis(f"  - {q}")
        tulis(f"    kunci: {kunci}")
        tulis(f"    top-1 yang terambil: {hasil[nama]['top1'][i][:150].strip()!r} ...")

# ------------------------------------------- selisih antar ukuran per pertanyaan
if len(KONFIG) >= 2:
    a, b = KONFIG[0][0], KONFIG[1][0]
    tulis(f"\n=== Pertanyaan yang hasilnya beda antara {a} dan {b} ===")
    beda = False
    for i, (q, _) in enumerate(UJI):
        ra, rb = hasil[a]["rank"][i], hasil[b]["rank"][i]
        if ra != rb:
            beda = True
            tulis(f"  - {q}  | {a}: peringkat {ra}  | {b}: peringkat {rb}")
    if not beda:
        tulis("  Tidak ada perbedaan.")

# ----------------------------------------------------- langkah 3: kerangka tulisan
tulis("""
=== LANGKAH 3: kerangka kesimpulan (isi sendiri dari angka di atas, 3-4 kalimat) ===
1. Ukuran chunk ___ memberi Hit@3 = ___ dibanding ___ untuk ukuran ___.
2. Pertanyaan yang gagal: ___ (sebut pertanyaan dan koleksinya).
3. Dugaan penyebab: ___ (misalnya chunk kecil memisahkan jawaban dari penjelasannya,
   chunk besar mencampur banyak topik, kata kunci ada di chunk tetapi bukan jawabannya,
   atau pertanyaan memakai istilah berbeda dari dokumen).
4. Keterbatasan: evaluasi mencocokkan kata kunci, jadi chunk yang memuat kata kunci
   belum tentu benar-benar menjawab. Periksa sebagian hasil secara manual.
""")

Path("lab3_hasil.txt").write_text("\n".join(LAPORAN), encoding="utf-8")
print("Hasil disimpan ke lab3_hasil.txt")
