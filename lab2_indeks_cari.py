"""
Lab 2 - Bangun Indeks & Cari (25 menit)
P-10 RAG Part 1: Embedding & Vector Database

Langkah di modul:
1. Letakkan 2-3 PDF di folder dokumen/, jalankan skrip indeks, pastikan tiap file
   melaporkan jumlah chunk.
2. Ajukan 5 pertanyaan dengan query(). Untuk tiap pertanyaan, periksa apakah
   chunk teratas benar-benar relevan.
3. Ajukan 2 pertanyaan dengan kata berbeda dari teks dokumen (parafrase).
   Apakah pencarian semantik tetap menemukannya?

Cara pakai:
    pip install chromadb sentence-transformers pypdf langchain-text-splitters
    1) Isi PERTANYAAN dan PARAFRASE di bawah dengan pertanyaan dari PDF Anda.
    2) python lab2_indeks_cari.py

Indeks disimpan di ./chroma_db (koleksi "pedoman") dan dipakai lagi di P-11.
"""
import sys
from pathlib import Path

from pypdf import PdfReader

# ChromaDB + SentenceTransformer digunakan jika tersedia.
# Jika dependensi belum terpasang, skrip tetap dapat diuji dengan pencarian lokal.
try:
    import chromadb
    from chromadb.utils import embedding_functions
    MODE_CHROMA = True
except ModuleNotFoundError:
    chromadb = None
    embedding_functions = None
    MODE_CHROMA = False

# ---------------------------------------------------------------- pengaturan
FOLDER = Path("dokumen")
UKURAN, TUMPANG = 800, 100        # nilai bawaan modul
N_HASIL = 3
NILAI_MANUAL = False              # False: hasil ditinjau tanpa input penilaian manual

# Langkah 2: 5 pertanyaan yang jawabannya ADA di PDF Anda, dengan kata yang mirip teks.
# Dua contoh pertama dari modul; ganti semuanya sesuai dokumen Anda.
PERTANYAAN = [
    "Apa syarat pengajuan cuti akademik?",
    "Dalam skenario dokumen, berapa minimal 75% kehadiran dari pertemuan terjadwal untuk mengikuti ujian akhir semester?",
    "Apa perbedaan Indeks Prestasi Semester (IPS) dan Indeks Prestasi Kumulatif (IPK)?",
    "Apa kegunaan Kartu Hasil Studi (KHS) untuk melihat nilai mata kuliah per semester?",
    "Apa langkah yang harus dilakukan untuk menjaga keamanan akun akademik dari serangan phishing?",
]

# Langkah 3: 2 pertanyaan yang artinya sama dengan isi dokumen, tetapi kata-katanya
# sengaja BERBEDA dari teks dokumen (parafrase).
PARAFRASE = [
    "Jika mahasiswa perlu tidak mengikuti perkuliahan selama satu semester, bagaimana cara mengajukan izin resmi?",
    "Bagaimana cara mengenali tautan atau pesan palsu yang berusaha mencuri informasi login mahasiswa?",
]

# Kata kunci utama yang diambil dari isi ketiga PDF referensi.
KATA_KUNCI = [
    "cuti akademik, dua semester aktif, formulir, dosen wali, program studi",
    "kehadiran, 75 persen, ujian akhir semester, pertemuan terjadwal",
    "IPS, IPK, SKS, evaluasi hasil belajar",
    "Kartu Hasil Studi (KHS), nilai mata kuliah, semester",
    "keamanan akun, kata sandi unik, autentikasi multifaktor, OTP, phishing",
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


# ------------------------------------------------------------------- langkah 1
pdf_list = sorted(FOLDER.glob("*.pdf"))
if not pdf_list:
    sys.exit(f"Tidak ada PDF di folder '{FOLDER}/'. Letakkan 2-3 PDF berbasis teks di sana.")

if MODE_CHROMA:
    # Model embedding multilingual (jalan lokal, tanpa API key)
    emb = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2")
    client = chromadb.PersistentClient(path="./chroma_db")
    koleksi = client.get_or_create_collection(
        name="pedoman", embedding_function=emb,
        metadata={"hnsw:space": "cosine"})
else:
    # Fallback tanpa instalasi tambahan: pencarian kemiripan kata berbasis cosine.
    import math, re
    class LocalCollection:
        def __init__(self):
            self.data = {}
        def upsert(self, ids, documents, metadatas):
            for i, d, m in zip(ids, documents, metadatas):
                self.data[i] = (d, m)
        def count(self):
            return len(self.data)
        def query(self, query_texts, n_results):
            q_tokens = set(re.findall(r"[a-zA-Z0-9]+", query_texts[0].lower()))
            scored = []
            for doc, meta in self.data.values():
                tokens = re.findall(r"[a-zA-Z0-9]+", doc.lower())
                d_tokens = set(tokens)
                overlap = len(q_tokens & d_tokens)
                # Jaccard-like score; distance makin kecil berarti makin mirip.
                score = overlap / max(1, len(q_tokens))
                scored.append((score, doc, meta))
            scored.sort(key=lambda x: x[0], reverse=True)
            chosen = scored[:n_results]
            return {"documents": [[x[1] for x in chosen]],
                    "metadatas": [[x[2] for x in chosen]],
                    "distances": [[1.0-x[0] for x in chosen]]}
    koleksi = LocalCollection()

tulis("=== LANGKAH 1: Indexing ===")
tulis("Mode pencarian: " + ("ChromaDB + SentenceTransformer" if MODE_CHROMA else "Fallback lokal berbasis kemiripan kata (bukan embedding semantik)"))
for pdf in pdf_list:
    teks = baca_pdf(pdf)
    chunks = potong(teks, UKURAN, TUMPANG)
    if not chunks:
        tulis(f"{pdf.name} -> 0 chunk (teks kosong, kemungkinan PDF hasil scan. Lewati.)")
        continue
    koleksi.upsert(                                            # upsert: aman dijalankan ulang
        ids=[f"{pdf.stem}-{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"sumber": pdf.name, "chunk": i} for i in range(len(chunks))],
    )
    tulis(f"{pdf.name} -> {len(chunks)} chunk")
tulis(f"Total isi koleksi 'pedoman': {koleksi.count()} chunk\n")


# ------------------------------------------------------------- fungsi pencarian
def cari(pertanyaan):
    hasil = koleksi.query(query_texts=[pertanyaan], n_results=N_HASIL)
    return list(zip(hasil["documents"][0], hasil["metadatas"][0], hasil["distances"][0]))


def tanya_relevan():
    if not NILAI_MANUAL:
        return None
    while True:
        j = input("Chunk teratas relevan dan cukup untuk menjawab? (y/n): ").strip().lower()
        if j in ("y", "n"):
            return j == "y"


def jalankan(daftar, judul):
    tulis(f"=== {judul} ===")
    penilaian = []
    for no, q in enumerate(daftar, 1):
        tulis(f"\n[{no}] Pertanyaan: {q}")
        if judul.startswith("LANGKAH 2"):
            tulis(f"  Kata kunci: {KATA_KUNCI[no - 1]}")
        for rank, (doc, meta, jarak) in enumerate(cari(q), 1):
            tulis(f"  #{rank} [{meta['sumber']} · chunk {meta['chunk']}] jarak={jarak:.3f}")
            tulis("     " + doc[:300].replace("\n", " ") + " ...")
        ok = tanya_relevan()
        if ok is not None:
            penilaian.append((q, ok))
            tulis(f"  Penilaian chunk teratas: {'RELEVAN' if ok else 'TIDAK RELEVAN'}")
    return penilaian


# ------------------------------------------------------------ langkah 2 dan 3
nilai_biasa = jalankan(PERTANYAAN, "LANGKAH 2: 5 pertanyaan biasa")
tulis()
nilai_parafrase = jalankan(PARAFRASE, "LANGKAH 3: 2 pertanyaan parafrase")

tulis("\n=== RINGKASAN ===")
if NILAI_MANUAL:
    b = sum(ok for _, ok in nilai_biasa)
    p = sum(ok for _, ok in nilai_parafrase)
    tulis(f"Pertanyaan biasa : chunk teratas relevan {b}/{len(nilai_biasa)}")
    tulis(f"Pertanyaan parafrase: chunk teratas relevan {p}/{len(nilai_parafrase)}")
    gagal = [q for q, ok in nilai_biasa + nilai_parafrase if not ok]
    if gagal:
        tulis("Pertanyaan yang gagal (analisis penyebabnya untuk laporan):")
        for q in gagal:
            tulis(f"  - {q}")
else:
    tulis("NILAI_MANUAL = False, jadi periksa relevansi hasil di atas dengan membaca sendiri.")

Path("lab2_hasil.txt").write_text("\n".join(LAPORAN), encoding="utf-8")
print("\nHasil disimpan ke lab2_hasil.txt")
