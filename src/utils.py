"""
utils.py
Fungsi bantu untuk visualisasi graph secara interaktif (pyvis) dan
helper umum lain yang dipakai lintas notebook/script.
"""

import networkx as nx


def visualize_pyvis(
    G: nx.Graph,
    partition: dict = None,
    out_path: str = "results/figures/aspect_network.html",
    height: str = "700px",
):
    """
    Buat visualisasi graph interaktif pakai pyvis, warna node berdasarkan
    komunitas hasil Louvain (jika `partition` diberikan).
    """
    from pyvis.network import Network
    import os

    net = Network(height=height, width="100%", notebook=False, bgcolor="#ffffff")
    net.barnes_hut()

    palette = [
        "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
        "#8338ec", "#ff006e", "#3a86ff", "#06d6a0", "#ffbe0b",
    ]

    for node in G.nodes():
        size = 15 + G.degree(node, weight="weight")
        color = palette[partition[node] % len(palette)] if partition else "#457b9d"
        net.add_node(node, label=node, size=size, color=color)

    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1)
        neg_ratio = data.get("neg_ratio", 0)
        title = f"co-occurrence: {weight}, neg_ratio: {neg_ratio}"
        net.add_edge(u, v, value=weight, title=title)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    net.write_html(out_path)
    print(f"Visualisasi interaktif tersimpan di {out_path}")


def summarize_graph(G: nx.Graph) -> dict:
    """Ringkasan statistik dasar graph, berguna untuk laporan."""
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": round(nx.density(G), 4),
        "avg_degree": round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 2)
        if G.number_of_nodes() > 0 else 0,
    }
