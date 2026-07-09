"""
run_pipeline.py
One-shot pipeline runner: preprocessing → ABSA → graph → visualize.

Jalankan dari root project:
    python run_pipeline.py

Flags (opsional):
    --skip-absa   : Skip tahap ABSA (gunakan absa_results.json yang sudah ada)
    --sample N    : Jumlah sampel acak untuk ABSA (default: 500)
"""

import argparse
import os
import sys
import time

# ─── Parse args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="SNA-ABSA E-Commerce Pipeline Runner")
parser.add_argument("--skip-absa", action="store_true",
                    help="Skip ABSA inference (pakai file absa_results.json yg sudah ada)")
parser.add_argument("--sample", type=int, default=500,
                    help="Jumlah sampel acak untuk ABSA inference (default: 500)")
args = parser.parse_args()

STEP = "[PIPELINE]"

def banner(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def check_or_exit(path, msg):
    if not os.path.exists(path):
        print(f"[ERROR] {msg}")
        sys.exit(1)

# ─── Step 0: Cek file mentah ada ─────────────────────────────────────────────
banner("Step 0: Cek Dataset")
RAW_CANDIDATES = [
    "data/raw/tokopedia-product-reviews-2019.csv",
    "tokopedia-product-reviews-2019.csv",
]
raw_path = None
for p in RAW_CANDIDATES:
    if os.path.exists(p):
        raw_path = p
        print(f"{STEP} Dataset ditemukan: {raw_path}")
        break

if raw_path is None:
    print("[ERROR] Dataset tidak ditemukan. Taruh file CSV di data/raw/ terlebih dahulu.")
    sys.exit(1)

# ─── Step 1: Preprocessing ───────────────────────────────────────────────────
banner("Step 1: Preprocessing")
t0 = time.time()
import pandas as pd
import sys
sys.path.insert(0, "src")
from preprocessing import preprocess_dataframe

os.makedirs("data/processed", exist_ok=True)
CLEAN_PATH = "data/processed/reviews_clean.csv"

df = pd.read_csv(raw_path)
print(f"{STEP} Raw baris: {len(df):,}")
df_clean = preprocess_dataframe(df, text_col="text")
df_clean.to_csv(CLEAN_PATH, index=False)
print(f"{STEP} Setelah cleaning: {len(df_clean):,} baris -> {CLEAN_PATH}")
print(f"{STEP} Selesai dalam {time.time()-t0:.1f}s")

# ─── Step 2: ABSA ────────────────────────────────────────────────────────────
ABSA_PATH = "data/processed/absa_results.json"

if args.skip_absa:
    check_or_exit(ABSA_PATH, f"--skip-absa aktif tapi {ABSA_PATH} tidak ditemukan. Jalankan tanpa flag ini dulu.")
    print(f"\n{STEP} [SKIP] Menggunakan {ABSA_PATH} yang sudah ada.")
else:
    banner(f"Step 2: ABSA Inference (sampel: {args.sample} baris)")
    t0 = time.time()
    import json
    from absa_model import ABSAExtractor, batch_extract

    extractor = ABSAExtractor()
    sample = df_clean.sample(n=min(args.sample, len(df_clean)), random_state=42)
    results = batch_extract(sample, extractor=extractor)

    with open(ABSA_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"{STEP} {len(results)} entri disimpan -> {ABSA_PATH}")
    print(f"{STEP} Selesai dalam {time.time()-t0:.1f}s")

# ─── Step 3: Build Graph ─────────────────────────────────────────────────────
banner("Step 3: Build Graph & SNA")
t0 = time.time()
import json
from graph_builder import (build_cooccurrence_graph, compute_centrality,
                            detect_communities, export_gexf)
from utils import summarize_graph

with open(ABSA_PATH, encoding="utf-8") as f:
    absa_results = json.load(f)

G = build_cooccurrence_graph(absa_results)
summary = summarize_graph(G)
print(f"{STEP} Graph: {summary}")

centrality = compute_centrality(G)
partition  = detect_communities(G)
print(f"{STEP} Komunitas terdeteksi: {len(set(partition.values()))}")

# Export GEXF
export_gexf(G)

# Export centrality CSV
import pandas as pd
os.makedirs("results", exist_ok=True)
df_c = pd.DataFrame(centrality).round(4)
df_c.index.name = "Aspek"
df_c.to_csv("results/centrality_metrics.csv")

# Export edge attributes CSV
edges_data = []
for u, v, d in G.edges(data=True):
    edges_data.append({
        "Aspek_1": u, "Aspek_2": v,
        "co_occurrence": d.get("weight", 0),
        "neg_count": d.get("neg_alignment", 0),
        "neg_ratio": d.get("neg_ratio", 0)
    })
pd.DataFrame(edges_data).to_csv("results/edge_attributes.csv", index=False)
print(f"{STEP} Selesai dalam {time.time()-t0:.1f}s")

# ─── Step 4: Visualize ───────────────────────────────────────────────────────
banner("Step 4: Generate Figures")
t0 = time.time()
from visualize import (plot_sentiment_distribution, plot_aspect_frequency,
                       plot_aspect_sentiment_breakdown, plot_cooccurrence_heatmap,
                       plot_neg_alignment_heatmap, plot_confidence_distribution, load_data)
from graph_builder import visualize_graph

os.makedirs("results/figures", exist_ok=True)
data = load_data()
plot_sentiment_distribution(data, "results/figures")
plot_aspect_frequency(data, "results/figures")
plot_aspect_sentiment_breakdown(data, "results/figures")
plot_cooccurrence_heatmap(data, "results/figures")
plot_neg_alignment_heatmap(data, "results/figures")
plot_confidence_distribution(data, "results/figures")
visualize_graph(G, partition)

print(f"{STEP} Selesai dalam {time.time()-t0:.1f}s")

# ─── Ringkasan Akhir ─────────────────────────────────────────────────────────
banner("PIPELINE SELESAI !")
import glob
files = [f for f in glob.glob("results/**/*", recursive=True) if os.path.isfile(f)]
print(f"\nTotal file output dihasilkan: {len(files)}")
for fpath in sorted(files):
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {fpath:<55} ({size_kb:.1f} KB)")

print()
print("Langkah selanjutnya:")
print("  1. Buka notebooks/01_eda.ipynb untuk eksplorasi data")
print("  2. Buka results/figures/aspect_network.html untuk visualisasi interaktif")
print("  3. Buka results/graph_exports/aspect_network.gexf di Gephi")
