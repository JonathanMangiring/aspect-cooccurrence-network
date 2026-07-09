"""
preprocessing.py
Membersihkan dan menormalisasi teks ulasan Bahasa Indonesia sebelum
masuk ke tahap ABSA.

Langkah:
1. Case folding (lowercase)
2. Hapus URL, mention, karakter non-alfanumerik berlebih
3. Normalisasi kata gaul/singkatan umum (kamus kecil, bisa diperluas)
4. (Opsional) Hapus stopword — untuk ABSA berbasis transformer biasanya
   TIDAK perlu hapus stopword karena model butuh konteks kalimat utuh.
   Stopword removal hanya dipakai kalau nanti butuh pendekatan lexicon-based.
"""

import re
import pandas as pd

# Kamus normalisasi kecil, silakan diperluas sesuai temuan di data
SLANG_DICT = {
    "gak": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "bgt": "banget",
    "bgt.": "banget",
    "yg": "yang",
    "dgn": "dengan",
    "utk": "untuk",
    "tdk": "tidak",
    "sm": "sama",
    "brp": "berapa",
    "recommended": "direkomendasikan",
    "recomended": "direkomendasikan",
    "ori": "original",
    "ori.": "original",
    "seller": "penjual",
    "cod": "bayar di tempat",
    "fast respon": "respon cepat",
    "fast response": "respon cepat",
}


def clean_text(text: str) -> str:
    """Basic cleaning: lowercase, hapus URL, hapus karakter berulang berlebih."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URL
    text = re.sub(r"[^a-z0-9\s.,!?]", " ", text)            # simbol aneh/emoji
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)               # "mantaaap" -> "mantaap"
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_slang(text: str, slang_dict: dict = None) -> str:
    """Ganti kata gaul/singkatan umum dengan bentuk baku sederhana."""
    slang_dict = slang_dict or SLANG_DICT
    words = text.split()
    normalized = [slang_dict.get(w, w) for w in words]
    return " ".join(normalized)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    min_length: int = 3,
) -> pd.DataFrame:
    """
    Terapkan cleaning + normalisasi ke seluruh kolom teks, lalu buang
    baris yang teksnya terlalu pendek (tidak informatif untuk ABSA).
    """
    df = df.copy()
    df[text_col] = df[text_col].apply(clean_text)
    df[text_col] = df[text_col].apply(normalize_slang)
    df = df[df[text_col].str.split().str.len() >= min_length]
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    import os

    RAW_CANDIDATES = [
        "data/raw/tokopedia-product-reviews-2019.csv",
        "tokopedia-product-reviews-2019.csv"
    ]
    RAW_PATH = None
    for p in RAW_CANDIDATES:
        if os.path.exists(p):
            RAW_PATH = p
            break

    OUT_PATH = "data/processed/reviews_clean.csv"

    if RAW_PATH is None:
        print("[!] File tidak ditemukan di data/raw/ atau root project.")
        print("    Download dataset dari Kaggle dan taruh di data/raw/ terlebih dahulu.")
    else:
        df = pd.read_csv(RAW_PATH)
        print(f"Loaded {len(df)} baris dari {RAW_PATH}")

        df_clean = preprocess_dataframe(df, text_col="text")
        print(f"Setelah cleaning: {len(df_clean)} baris tersisa")

        os.makedirs("data/processed", exist_ok=True)
        df_clean.to_csv(OUT_PATH, index=False)
        print(f"Tersimpan di {OUT_PATH}")
