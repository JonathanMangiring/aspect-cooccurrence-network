# Laporan Temuan: Aspect Co-occurrence Network pada Ulasan E-Commerce

## 1. Ringkasan Dataset
- Sumber: Tokopedia Product Reviews (Kaggle)
- Jumlah ulasan (sebelum/sesudah cleaning): 40.607 / 33.968
- Jumlah ulasan yang mengandung aspek relevan (sampel 500): 339
- Kategori produk yang dicakup: E-commerce umum (berbagai kategori)

## 2. Metodologi Singkat
1. Preprocessing teks (cleaning, normalisasi slang)
2. Ekstraksi aspek (keyword-based) — aspek yang dipakai: harga, kualitas, pengiriman, packing, pelayanan
3. Klasifikasi sentimen per ulasan yang menyebut aspek tsb (IndoBERT ABSA — `damasukma/indobert-absa`)
4. Membangun aspect co-occurrence graph (node = aspek, edge = ko-kemunculan)
5. Community detection (Louvain) + centrality analysis
6. Visualisasi: static PNG (matplotlib/seaborn) + interaktif HTML (pyvis)

## 3. Temuan Utama

### 3.1 Statistik Graph
- Jumlah node (aspek): 5
- Jumlah edge: 10
- Density: 1.0 (graph komplit — semua aspek terhubung satu sama lain)

### 3.2 Aspek Paling Sentral
| Aspek | Degree Centrality |
|-------|-------------------|
| pengiriman | 1.000 |
| pelayanan | 1.000 |
| kualitas | 1.000 |
| packing | 1.000 |
| harga | 1.000 |

*Interpretasi: aspek dengan centrality tinggi adalah aspek yang paling sering
dikaitkan dengan aspek lain — indikasi bahwa aspek ini paling "menentukan"
persepsi keseluruhan produk.*

### 3.3 Komunitas Aspek yang Terdeteksi
- Jumlah Komunitas: 1
- Komunitas 1: [pengiriman, pelayanan, kualitas, packing, harga] — interpretasi:
  Seluruh aspek saling terkait erat satu sama lain membentuk satu komunitas besar.
  Pembeli di Tokopedia cenderung membicarakan kelima aspek ini secara bersamaan
  tanpa terpisah menjadi sub-grup spesifik.

### 3.4 Pasangan Aspek dengan Co-occurrence Terbanyak
| Aspek 1 | Aspek 2 | Co-occurrence |
|---------|---------|--------------|
| pengiriman | kualitas | 43 |
| pengiriman | pelayanan | 36 |
| pelayanan | kualitas | 31 |
| kualitas | harga | 18 |
| pengiriman | packing | 12 |

## 4. Visualisasi

| File | Deskripsi |
|------|-----------|
| `results/figures/aspect_network.png` | Graph co-occurrence statik (node ukuran = sentralitas, lebar edge = frekuensi) |
| `results/figures/aspect_network.html` | Graph co-occurrence interaktif (drag, zoom, hover tooltip) |
| `results/figures/sentiment_distribution.png` | Pie chart distribusi sentimen (positif/netral/negatif) |
| `results/figures/aspect_frequency.png` | Bar chart frekuensi kemunculan per aspek |
| `results/figures/aspect_sentiment_breakdown.png` | Stacked bar sentimen per aspek |
| `results/figures/cooccurrence_heatmap.png` | Heatmap frekuensi co-occurrence antar aspek |
| `results/figures/confidence_distribution.png` | Histogram confidence score prediksi model |

> Untuk meregenerasi semua figure: `python src/visualize.py`

## 5. Studi Banding Antar Kategori Produk (opsional)
(Akan dianalisis jika data memiliki pemisahan kategori yang jelas.)

## 6. Keterbatasan
- Model ABSA (`damasukma/indobert-absa`) di-fine-tune pada domain review
  makanan; performa pada domain lain perlu divalidasi lebih lanjut.
- Ekstraksi aspek berbasis keyword rentan miss pada ekspresi implisit
  (misal "kirimnya lama banget" tanpa kata "pengiriman").
- Graph co-occurrence bersifat undirected & tidak menangkap urutan/kausalitas.
- Analisis dijalankan pada sampel acak 500 ulasan (dari 33.968) untuk efisiensi CPU.

## 7. Kesimpulan
- **Semua aspek saling terhubung kuat (Density 1.0)**: Pembeli di Tokopedia cenderung
  mengaitkan kualitas produk, pengiriman, pelayanan, kemasan, dan harga secara bersamaan.
  Tidak ada aspek yang benar-benar terisolasi.
- **Sentralitas Merata**: Kelima aspek memiliki degree centrality yang tinggi (1.000),
  menunjukkan bahwa kepuasan pelanggan e-commerce sangat holistik.
- **Pasangan Dominan**: Pengiriman & kualitas adalah pasangan aspek yang paling sering
  disebut bersamaan (co-occurrence = 43), diikuti pengiriman & pelayanan (36).

## 8. Referensi
- Model: damasukma/indobert-absa — https://huggingface.co/damasukma/indobert-absa
- Dataset: Tokopedia Product Reviews — https://www.kaggle.com/datasets/farhan999/tokopedia-product-reviews
