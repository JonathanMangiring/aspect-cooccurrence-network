"""
absa_model.py
Wrapper untuk melakukan inference Aspect-Based Sentiment Analysis (ABSA)
menggunakan model IndoBERT yang sudah di-fine-tune.

Model default: damasukma/indobert-absa (HuggingFace)
  - Aspek contoh: rasa, harga, kualitas, pengiriman
  - Output: sentimen per aspek (positive / neutral / negative)

CATATAN PENTING:
Model ini di-fine-tune pada domain review MAKANAN. Untuk domain marketplace
umum (elektronik, fashion, dll), sebaiknya:
  1. Evaluasi dulu kualitas prediksinya secara kualitatif (sampling manual)
  2. Sesuaikan/tambah daftar ASPECT_LABELS di bawah sesuai kategori produkmu
  3. Jika akurasi kurang, fine-tune ulang dari indobenchmark/indobert-base-p1
     dengan data berlabel dari domain e-commerce (lihat docs/report.md
     untuk catatan rencana fine-tuning)

Jika model belum mendukung aspek yang kamu butuhkan secara langsung,
pendekatan alternatif yang lebih robust untuk domain umum adalah:
  - Aspect extraction via keyword/lexicon matching (daftar kata kunci per
    aspek: harga, kualitas, pengiriman, packing, pelayanan)
  - Sentiment classification per kalimat yang mengandung aspek tsb, pakai
    model sentiment umum (misal indobert sentiment/SmSA-based)
Fungsi `extract_aspects_keyword()` di bawah menyediakan fallback ini.
"""

from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_NAME = "damasukma/indobert-absa"

# Aspek default untuk domain e-commerce umum (dipakai fallback keyword-based)
ASPECT_KEYWORDS = {
    "harga": ["harga", "murah", "mahal", "worth it", "worth", "terjangkau"],
    "kualitas": ["kualitas", "bagus", "jelek", "awet", "original", "kw", "rusak"],
    "pengiriman": ["kirim", "pengiriman", "paket", "sampai", "expedisi", "kurir", "lama"],
    "packing": ["packing", "bungkus", "kardus", "bubble wrap", "pecah"],
    "pelayanan": ["respon", "seller", "penjual", "ramah", "fast respon", "cs"],
}


class ABSAExtractor:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model {model_name} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Dict:
        """
        Jalankan model pada satu teks ulasan.
        NOTE: sesuaikan post-processing ini dengan format output asli model
        (cek `model.config.id2label` setelah load, karena format multi-aspek
        bisa berbeda-beda antar model fine-tuned di HuggingFace).
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_id = int(torch.argmax(probs, dim=-1))

        raw_label = self.model.config.id2label.get(pred_id, str(pred_id))
        label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
        label = label_map.get(raw_label, raw_label)
        
        confidence = float(probs[0][pred_id])

        return {"label": label, "confidence": confidence}


def extract_aspects_keyword(text: str, aspect_keywords: dict = None) -> List[str]:
    """
    Fallback sederhana: deteksi aspek mana saja yang disebut dalam teks
    berdasarkan kemunculan kata kunci. Berguna sebagai langkah AWAL sebelum
    menjalankan model sentiment per-aspek, atau sebagai baseline pembanding.
    """
    aspect_keywords = aspect_keywords or ASPECT_KEYWORDS
    found = []
    text_lower = text.lower()
    for aspect, keywords in aspect_keywords.items():
        if any(kw in text_lower for kw in keywords):
            found.append(aspect)
    return found


def batch_extract(df, text_col: str = "text", extractor: ABSAExtractor = None):
    """
    Jalankan ekstraksi aspek (keyword-based) + sentimen (model-based) untuk
    seluruh DataFrame. Mengembalikan list of dict:
    [{review_id, aspects: [...], sentiment_label, confidence}, ...]
    """
    from tqdm import tqdm
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting ABSA"):
        text = row[text_col]
        aspects = extract_aspects_keyword(text)

        if not aspects:
            continue  # skip review yang tidak menyebut aspek yang kita track

        entry = {"review_id": idx, "text": text, "aspects": aspects}

        if extractor is not None:
            pred = extractor.predict(text)
            entry["sentiment_label"] = pred["label"]
            entry["confidence"] = pred["confidence"]

        results.append(entry)

        # Bebaskan memori secara eksplisit untuk mencegah OOM
        import gc
        gc.collect()

    return results


if __name__ == "__main__":
    import pandas as pd
    import json
    import os

    IN_PATH = "data/processed/reviews_clean.csv"
    OUT_PATH = "data/processed/absa_results.json"

    if not os.path.exists(IN_PATH):
        print(f"[!] File tidak ditemukan: {IN_PATH}. Jalankan preprocessing.py dulu.")
    else:
        df = pd.read_csv(IN_PATH)

        # Cek dulu label yang tersedia sebelum full-run (penting untuk validasi model)
        extractor = ABSAExtractor()
        print("Label asli dari model:", extractor.model.config.id2label)

        print("Mengekstraksi sampel acak 500 baris agar komputasi di CPU cepat dan representatif...")
        sample = df.sample(n=min(500, len(df)), random_state=42)
        
        results = batch_extract(sample, extractor=extractor)
        
        # Simpan hasil ekstraksi agar bisa digunakan oleh graph_builder.py
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"Selesai! Hasil disimpan di {OUT_PATH}")
