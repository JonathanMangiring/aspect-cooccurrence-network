"""
graph_builder.py
Membangun Aspect Co-occurrence Network dari hasil ekstraksi ABSA.

Konsep graph:
- Node   = aspek (harga, kualitas, pengiriman, packing, pelayanan, ...)
- Edge   = muncul bersamaan dalam review yang sama
- Bobot edge = kombinasi dari:
    (a) frekuensi co-occurrence, dan
    (b) "keselarasan sentimen negatif" -- seberapa sering kedua aspek itu
        sama-sama negatif di review yang sama (mengindikasikan satu aspek
        buruk "menyeret" persepsi aspek lain)

Ini yang membuat pendekatan SNA di project ini beda dari SNA biasa (yang
biasanya berbasis user-to-user network).
"""

from itertools import combinations
from collections import defaultdict
import networkx as nx


def build_cooccurrence_graph(absa_results: list) -> nx.Graph:
    """
    absa_results: list of dict, masing-masing punya keys:
        - aspects: List[str]
        - sentiment_label: str ('positive' / 'neutral' / 'negative')

    Return: networkx.Graph dengan atribut edge:
        - weight: jumlah co-occurrence
        - neg_alignment: jumlah co-occurrence saat sentimen sama-sama negatif
    """
    G = nx.Graph()
    pair_counts = defaultdict(int)
    neg_alignment_counts = defaultdict(int)

    for entry in absa_results:
        aspects = entry.get("aspects", [])
        sentiment = entry.get("sentiment_label", "neutral")

        for aspect in aspects:
            G.add_node(aspect)

        for a1, a2 in combinations(sorted(set(aspects)), 2):
            pair_counts[(a1, a2)] += 1
            if sentiment == "negative":
                neg_alignment_counts[(a1, a2)] += 1

    for (a1, a2), count in pair_counts.items():
        neg_count = neg_alignment_counts.get((a1, a2), 0)
        G.add_edge(
            a1, a2,
            weight=count,
            neg_alignment=neg_count,
            neg_ratio=round(neg_count / count, 3) if count else 0,
        )

    return G


def compute_centrality(G: nx.Graph) -> dict:
    """Hitung beberapa metrik centrality standar untuk identifikasi aspek paling berpengaruh."""
    return {
        "degree": nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G, weight="weight"),
        "eigenvector": nx.eigenvector_centrality(G, weight="weight", max_iter=1000),
    }


def detect_communities(G: nx.Graph):
    """
    Community detection pakai algoritma Louvain.
    Return: dict {node: community_id}
    """
    try:
        import community as community_louvain  # python-louvain
    except ImportError:
        raise ImportError(
            "Package 'python-louvain' belum terinstall. Jalankan: pip install python-louvain"
        )

    partition = community_louvain.best_partition(G, weight="weight")
    return partition


def export_gexf(G: nx.Graph, path: str = "results/graph_exports/aspect_network.gexf"):
    """Ekspor graph ke format .gexf supaya bisa dibuka & dieksplorasi visual di Gephi."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nx.write_gexf(G, path)
    print(f"Graph diekspor ke {path}")


def visualize_graph(G: nx.Graph, partition: dict, path_static: str = "results/figures/aspect_network.png", path_interactive: str = "results/figures/aspect_network.html"):
    """Buat visualisasi statik dan interaktif untuk graph."""
    import os
    import matplotlib.pyplot as plt
    from pyvis.network import Network
    import matplotlib.colors as mcolors

    os.makedirs(os.path.dirname(path_static), exist_ok=True)
    
    # 1. Visualisasi Statik (Matplotlib)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    colors = list(mcolors.TABLEAU_COLORS.values())
    node_colors = [colors[partition[n] % len(colors)] for n in G.nodes()]
    deg_centrality = nx.degree_centrality(G)
    node_sizes = [3000 * deg_centrality[n] + 1000 for n in G.nodes()]
    
    edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [1 + 5 * (w / max_w) for w in edge_weights]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, edgecolors="white", linewidths=1.5)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif", font_weight="bold")

    plt.title("Aspect Co-occurrence Network\n(Ukuran node = Sentralitas, Lebar edge = Frekuensi Co-occurrence)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path_static, dpi=300)
    plt.close()
    print(f"Visualisasi statik diekspor ke {path_static}")

    # 2. Visualisasi Interaktif (Pyvis)
    try:
        net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
        
        # Pyvis gak selalu support atribut graph networkx kompleks langsung,
        # jadi kita copy dan set node/edge spesifik
        for n in G.nodes():
            net.add_node(n, label=n, title=f"Aspek: {n}\nSentralitas: {deg_centrality[n]:.2f}", 
                         group=partition[n], value=deg_centrality[n] * 50)
                         
        for u, v, d in G.edges(data=True):
            w = d.get('weight', 1)
            net.add_edge(u, v, value=w, title=f"Co-occurrence: {w}")
            
        net.save_graph(path_interactive)
        print(f"Visualisasi interaktif diekspor ke {path_interactive}")
    except Exception as e:
        print(f"Gagal membuat visualisasi interaktif (pyvis mungkin belum terinstall dengan baik): {e}")



if __name__ == "__main__":
    import json
    import os

    IN_PATH = "data/processed/absa_results.json"

    if not os.path.exists(IN_PATH):
        print(f"[!] File tidak ditemukan: {IN_PATH}. Jalankan absa_model.py dulu (full run).")
    else:
        with open(IN_PATH) as f:
            absa_results = json.load(f)

        G = build_cooccurrence_graph(absa_results)
        print(f"Graph terbentuk: {G.number_of_nodes()} node, {G.number_of_edges()} edge")

        centrality = compute_centrality(G)
        print("\nTop 5 aspek berdasarkan degree centrality:")
        top5 = sorted(centrality["degree"].items(), key=lambda x: -x[1])[:5]
        for node, score in top5:
            print(f"  {node}: {score:.3f}")

        partition = detect_communities(G)
        print("\nJumlah komunitas terdeteksi:", len(set(partition.values())))

        export_gexf(G)
        visualize_graph(G, partition)

