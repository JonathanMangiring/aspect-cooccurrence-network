"""
visualize.py
Menghasilkan berbagai figure analisis dari hasil ABSA dan graph.

Output figures:
1. sentiment_distribution.png       - Pie chart distribusi sentimen
2. aspect_frequency.png             - Bar chart frekuensi per aspek
3. aspect_sentiment_breakdown.png   - Stacked bar sentimen per aspek
4. cooccurrence_heatmap.png         - Heatmap frekuensi co-occurrence antar aspek
5. aspect_network.png               - Graph co-occurrence (diperbarui via graph_builder)
6. aspect_network.html              - Visualisasi interaktif pyvis
"""

import json
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import defaultdict
from itertools import combinations

# ── Konfigurasi ──────────────────────────────────────────────────────────────
IN_PATH   = "data/processed/absa_results.json"
OUT_DIR   = "results/figures"
PALETTE   = {"positive": "#4CAF50", "neutral": "#FFC107", "negative": "#F44336"}
ASPECTS   = ["harga", "kualitas", "pengiriman", "packing", "pelayanan"]

def load_data():
    with open(IN_PATH, encoding="utf-8") as f:
        return json.load(f)

# ── 1. Distribusi Sentimen ────────────────────────────────────────────────────
def plot_sentiment_distribution(data, out_dir):
    counts = defaultdict(int)
    for entry in data:
        counts[entry.get("sentiment_label", "neutral")] += 1

    labels  = list(counts.keys())
    values  = list(counts.values())
    colors  = [PALETTE.get(l, "#9E9E9E") for l in labels]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        values, labels=None, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
        at.set_color("white")

    ax.legend(
        wedges, [f"{l.capitalize()} ({v})" for l, v in zip(labels, values)],
        loc="lower center", bbox_to_anchor=(0.5, -0.05),
        ncol=len(labels), fontsize=11
    )
    ax.set_title("Distribusi Sentimen Ulasan\n(Aspect-Based Sentiment Analysis)", 
                 fontsize=14, fontweight="bold", pad=20)

    path = os.path.join(out_dir, "sentiment_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")

# ── 2. Frekuensi Aspek ────────────────────────────────────────────────────────
def plot_aspect_frequency(data, out_dir):
    freq = defaultdict(int)
    for entry in data:
        for asp in entry.get("aspects", []):
            freq[asp] += 1

    # Urutkan
    df = pd.DataFrame(list(freq.items()), columns=["Aspek", "Frekuensi"])
    df = df.sort_values("Frekuensi", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df["Aspek"], df["Frekuensi"],
                   color="#5C6BC0", edgecolor="white", linewidth=0.8)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1, bar.get_y() + bar.get_height() / 2,
                str(int(w)), va="center", ha="left", fontsize=11)

    ax.set_xlabel("Jumlah Ulasan yang Menyebut Aspek", fontsize=11)
    ax.set_title("Frekuensi Kemunculan Aspek dalam Ulasan", 
                 fontsize=13, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    path = os.path.join(out_dir, "aspect_frequency.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")

# ── 3. Breakdown Sentimen per Aspek ──────────────────────────────────────────
def plot_aspect_sentiment_breakdown(data, out_dir):
    counts = {asp: {"positive": 0, "neutral": 0, "negative": 0} for asp in ASPECTS}
    for entry in data:
        sent = entry.get("sentiment_label", "neutral")
        for asp in entry.get("aspects", []):
            if asp in counts:
                counts[asp][sent] += 1

    df = pd.DataFrame(counts).T
    df = df[["positive", "neutral", "negative"]]  # urutan kolom

    ax = df.plot(
        kind="bar", stacked=True, figsize=(10, 6),
        color=[PALETTE["positive"], PALETTE["neutral"], PALETTE["negative"]],
        edgecolor="white", linewidth=0.5
    )
    ax.set_xlabel("Aspek", fontsize=11)
    ax.set_ylabel("Jumlah Ulasan", fontsize=11)
    ax.set_title("Distribusi Sentimen per Aspek", fontsize=13, fontweight="bold")
    ax.legend(title="Sentimen", labels=["Positif", "Netral", "Negatif"],
              loc="upper right", fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
    ax.spines[["top","right"]].set_visible(False)

    path = os.path.join(out_dir, "aspect_sentiment_breakdown.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")

# ── 4. Heatmap Co-occurrence ──────────────────────────────────────────────────
def plot_cooccurrence_heatmap(data, out_dir):
    matrix = pd.DataFrame(0, index=ASPECTS, columns=ASPECTS)
    for entry in data:
        aspects = [a for a in entry.get("aspects", []) if a in ASPECTS]
        for a1, a2 in combinations(sorted(set(aspects)), 2):
            matrix.at[a1, a2] += 1
            matrix.at[a2, a1] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="YlOrRd",
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 12}
    )
    ax.set_title("Heatmap Frekuensi Co-occurrence Antar Aspek",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)

    path = os.path.join(out_dir, "cooccurrence_heatmap.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


# ── 4b. Neg-Alignment Heatmap ─────────────────────────────────────────────────
def plot_neg_alignment_heatmap(data, out_dir):
    """Heatmap neg_ratio per pasangan aspek."""
    import networkx as nx
    from graph_builder import build_cooccurrence_graph

    G = build_cooccurrence_graph(data)
    neg_matrix = pd.DataFrame(0.0, index=ASPECTS, columns=ASPECTS)
    for u, v, d in G.edges(data=True):
        if u in ASPECTS and v in ASPECTS:
            nr = d.get("neg_ratio", 0)
            neg_matrix.at[u, v] = nr
            neg_matrix.at[v, u] = nr

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        neg_matrix, annot=True, fmt=".2f", cmap="Reds",
        linewidths=0.5, linecolor="white",
        vmin=0, vmax=1,
        ax=ax, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 12}
    )
    ax.set_title("Neg-Alignment Ratio per Pasangan Aspek\n"
                 "(seberapa sering kedua aspek sama-sama negatif)",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)

    path = os.path.join(out_dir, "neg_alignment_heatmap.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


# ── 5. Confidence Score Distribution ─────────────────────────────────────────
def plot_confidence_distribution(data, out_dir):
    scores = [e.get("confidence", 0) for e in data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=20, color="#26A69A", edgecolor="white", linewidth=0.7)
    ax.axvline(pd.Series(scores).mean(), color="#E53935", linestyle="--",
               linewidth=2, label=f"Rata-rata: {pd.Series(scores).mean():.2f}")
    ax.set_xlabel("Confidence Score Model", fontsize=11)
    ax.set_ylabel("Jumlah Ulasan", fontsize=11)
    ax.set_title("Distribusi Confidence Score Prediksi ABSA",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines[["top","right"]].set_visible(False)

    path = os.path.join(out_dir, "confidence_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")

"""
visualize.py
Menghasilkan berbagai figure analisis dari hasil ABSA dan graph.

Output figures:
1. sentiment_distribution.png    - Pie chart distribusi sentimen
2. aspect_frequency.png          - Bar chart frekuensi per aspek
3. aspect_sentiment_breakdown.png- Stacked bar sentimen per aspek
4. cooccurrence_heatmap.png      - Heatmap frekuensi co-occurrence antar aspek
5. neg_alignment_heatmap.png     - Heatmap neg_ratio per pasangan aspek
6. confidence_distribution.png   - Histogram confidence score model
7. aspect_network.png            - Graph co-occurrence statik
8. aspect_network.html           - Visualisasi interaktif pyvis
"""

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(IN_PATH):
        print(f"[!] File tidak ditemukan: {IN_PATH}. Jalankan absa_model.py dulu.")
    else:
        os.makedirs(OUT_DIR, exist_ok=True)
        data = load_data()
        print(f"Loaded {len(data)} entries dari {IN_PATH}\n")

        plot_sentiment_distribution(data, OUT_DIR)
        plot_aspect_frequency(data, OUT_DIR)
        plot_aspect_sentiment_breakdown(data, OUT_DIR)
        plot_cooccurrence_heatmap(data, OUT_DIR)
        plot_neg_alignment_heatmap(data, OUT_DIR)
        plot_confidence_distribution(data, OUT_DIR)

        print("\nSemua figure tersimpan di results/figures/")

