# Dokumentasi: Model Sales Performance Analysis

Dokumen ini menjelaskan alur dari awal sampai akhir: menaruh dataset, cleaning dan merge, sampai training model regresi.

Referensi materi (folder `materi/`):
- `Cleaning.pdf` → data preparation (missing value, duplikat, tipe data)
- `Modeling.pdf` → alur train/test, data leakage
- `SUPERVISED LEARNING  REGRESSION.pdf` → Linear, Polynomial, Ridge, Lasso, metrik MAE/MSE/RMSE/R²

---

## 1. Struktur folder

```
model-sales-performance-analysis/
├── datasets/            # data RAW (input)
│   ├── customers.csv
│   ├── orders.csv
│   ├── payments.csv
│   ├── products.csv
│   └── clean_final_data.csv   # bukan raw; hasil referensi, TIDAK dipakai pipeline
├── materi/              # PDF materi kuliah
├── temp/                # semua OUTPUT pipeline
├── cleaning.py          # langkah 1: cleaning + merge
├── modeling.py          # langkah 2: split + training + evaluasi
└── DOKUMENTASI.md
```

## 2. Persiapan

### 2.1 Menaruh dataset
Letakkan 4 file raw di folder `datasets/` dengan nama persis seperti di atas:

| File | Isi | Kunci |
|---|---|---|
| `customers.csv` | Profil pelanggan (Age, City, SignupDate, CustomerSegment) | CustomerID |
| `orders.csv` | Transaksi (OrderDate, ProductID, Quantity, Discount, PaymentMethod, Status) | OrderID |
| `payments.csv` | Status pembayaran per order | OrderID |
| `products.csv` | Master produk (ProductName, Category, UnitPrice) | ProductID |

Folder `temp/` dibuat otomatis oleh `cleaning.py` jika belum ada.

### 2.2 Kebutuhan
Python 3 dengan `pandas` dan `scikit-learn`:

```
pip install pandas scikit-learn
```

### 2.3 Cara menjalankan
Jalankan dari root project, berurutan:

```
python cleaning.py     # menghasilkan temp/clean_merged.csv
python modeling.py     # menghasilkan split, metrik, dan tren per produk
```

---

## 3. Tahap 1: Cleaning dan merge (`cleaning.py`)

Mengikuti `Cleaning.pdf`. Prinsipnya: pilih teknik sesuai tujuan analisis. Data yang menentukan nominal transaksi tidak boleh diimputasi, tetapi baris dihapus bila missing sedikit (<5%) dan acak.

### 3.1 Masalah yang ditemukan di data raw

| Tabel | Masalah | Jumlah |
|---|---|---|
| orders | Baris duplikat penuh | 120 |
| orders | OrderDate kosong | 35 |
| orders | Quantity kosong | 80 |
| orders | Discount kosong | 221 |
| orders | PaymentMethod kosong | 452 |
| orders | Quantity ≤ 0 (tidak valid) | 25 |
| orders | CustomerID 999999 tidak ada di customers (orphan) | 30 |
| customers | Age kosong | 180 |
| customers | City kosong | 119 |
| customers | Penulisan City tidak seragam (`tehran`, `Mashad`) | 110 |
| payments | PaymentDate kosong | 35 |

### 3.2 Penanganan

| Masalah | Penanganan | Alasan |
|---|---|---|
| Duplikat | `drop_duplicates()` | OrderID unik per transaksi, baris identik penuh = input ganda |
| OrderDate / Quantity / Discount kosong | `dropna` pada baris tersebut | Menentukan waktu dan nominal Sales, jadi tidak boleh diimputasi (Kasus A materi). Persentasenya <1% dan acak. OrderDate yang kosong juga kosong di payments, jadi tidak bisa dipulihkan. |
| Quantity ≤ 0 | Baris dihapus | Pembelian minimal 1 unit, dianggap data invalid |
| Age kosong | Imputasi **median** | Numerik, median tidak sensitif outlier |
| City kosong | Isi **"Unknown"** | Memakai modus akan membiaskan ke Tehran |
| City tidak seragam | `str.title()` + `Mashad` → `Mashhad` | Menyamakan kategori yang sebenarnya sama |
| PaymentMethod kosong | Isi **"Unknown"** | Tidak ada sumber untuk menebak metode bayar |
| CustomerID orphan | Dibuang lewat inner join | Atribut pelanggan tidak ada |
| Tipe data | `pd.to_datetime`, `astype(int)` | Tanggal dan angka terbaca sebagai teks/float |

### 3.3 Merge

```
orders ──(OrderID, one_to_one)──> payments
       ──(ProductID, many_to_one)──> products
       ──(CustomerID, many_to_one, inner)──> customers
```

- Semua join memakai `validate=` agar tidak terjadi duplikasi baris (fan-out) tanpa disadari.
- `PaymentDate` dibuang karena identik dengan `OrderDate`.
- Kolom turunan: `Sales = Quantity × UnitPrice × (1 − Discount)`.

### 3.4 Output
`temp/clean_merged.csv`: **49.610 baris × 17 kolom**, tanpa missing value dan tanpa duplikat.

Kolom: OrderID, CustomerID, OrderDate, ProductID, Quantity, Discount, PaymentMethod, Status, PaymentStatus, Age, City, SignupDate, CustomerSegment, ProductName, Category, UnitPrice, Sales.

### 3.5 Catatan kualitas data (dicatat, tidak diubah)
- 13.892 order bertanggal sebelum `SignupDate` pelanggan.
- Ada order `Status = Completed` dengan `PaymentStatus = Failed` (1.752 baris) dan kombinasi tidak konsisten lain. Kedua kolom dibiarkan apa adanya.

---

## 4. Tahap 2: Training model (`modeling.py`)

Mengikuti `Modeling.pdf` dan `SUPERVISED LEARNING  REGRESSION.pdf`.

### 4.1 Tujuan dan bentuk data
- **Target (y):** Sales bulanan per produk. Hanya order `Status = Completed` yang dihitung. Bulan tanpa penjualan dicatat 0.
- **Fitur (X):** `t` = urutan bulan (0–29, Jan 2024 sampai Jun 2026).
- **Jenis masalah:** regresi (target berupa angka).
- **Satu model per produk** (20 produk), karena tiap produk punya pola tren sendiri.

### 4.2 Split 70 / 15 / 15

| Set | Fungsi | Bulan | Periode | Baris |
|---|---|---|---|---|
| Train | Belajar parameter | 21 | 2024-01 .. 2025-09 | 420 |
| Validasi | Memilih konfigurasi | 4 | 2025-10 .. 2026-01 | 80 |
| Test | Evaluasi akhir (sekali) | 5 | 2026-02 .. 2026-06 | 100 |

Split dilakukan **berurutan waktu**, bukan acak. Split acak membuat model melihat bulan "masa depan" saat training (data leakage, lihat `Modeling.pdf`). 30 bulan dibulatkan menjadi 21/4/5.

Hasil split disimpan sebagai `train_monthly.csv`, `validasi_monthly.csv`, `test_monthly.csv`.

### 4.3 Model yang dibandingkan
Semua dari keluarga regresi linear sesuai materi:

| Model | Keterangan |
|---|---|
| Naive (rata-rata train) | Hanya acuan: tebakan paling sederhana |
| **Linear Regression** | Baseline / model utama: ŷ = β₀ + β₁·t |
| Polynomial (derajat 2 dan 5) | Menambah t², …; diuji untuk melihat overfitting |
| Ridge (alpha 0,1; 1; 10) | Regularisasi L2, memakai `StandardScaler` |
| Lasso (alpha 0,1; 1; 10) | Regularisasi L1, memakai `StandardScaler` |

### 4.4 Metrik
MAE, MSE, RMSE, R² dihitung di train, validasi, dan test (gabungan semua produk).

### 4.5 Pemilihan model
1. Semua model dilatih di **train**.
2. Model dipilih berdasarkan **RMSE terkecil di validasi**.
3. **Test** dilaporkan sekali sebagai ukuran jujur pada data yang belum pernah dilihat.

### 4.6 Tren per produk
Linear Regression dilatih per produk. **Slope** (perubahan sales per bulan) menjadi penanda arah tren: negatif = turun. Disimpan dua versi di `temp/product_trend.csv`:
- `slope_train`: hanya data sebelum Okt 2025
- `slope_all`: seluruh data

### 4.7 Output di `temp/`

| File | Isi |
|---|---|
| `clean_merged.csv` | Data bersih hasil merge (dari tahap 1) |
| `train_monthly.csv`, `validasi_monthly.csv`, `test_monthly.csv` | Hasil split |
| `model_metrics.csv` | MAE, MSE, RMSE, R² tiap model di tiap set |
| `product_trend.csv` | Slope, R², dan label tren per produk |

---

## 5. Hasil dan keterbatasan

RMSE (makin kecil makin baik):

| Model | Train | Validasi | Test |
|---|---|---|---|
| Naive (rata-rata) | 1048 | 1110 | 2156 |
| Linear | 1022 | 1200 | 2262 |
| Polynomial 2 | 986 | 1496 | 3128 |
| Polynomial 5 | 877 | 10601 | 80687 |
| Ridge (alpha 10), terbaik di validasi | 1025 | 1150 | 2208 |

Kesimpulan dan keterbatasan:
- **Tidak ada model yang lebih baik dari tebakan rata-rata.** R² test Linear = −0,10 (lebih buruk dari menebak rata-rata).
- **Polynomial 5 overfitting:** error train kecil, tetapi error test meledak. Sesuai materi, model yang lebih fleksibel belum tentu lebih baik.
- **Penyebab utama:** sales stabil di 2024–2025 (±350–378 ribu per kuartal), lalu turun tajam mulai Feb 2026 (Q1 291 ribu, Q2 254 ribu; jumlah order juga turun). Model yang dilatih pada data stabil tidak dapat menebak penurunan mendadak ini. Perlu dipastikan apakah penurunan 2026 itu nyata atau data 2026 belum lengkap.
- **Slope sensitif terhadap periode:** dengan `slope_train` hanya 10 dari 20 produk yang turun, dengan `slope_all` semuanya turun. Kesimpulan "produk X selalu turun" belum bisa ditarik.
- **R² validasi tinggi (±0,85) menyesatkan**, karena dihitung gabungan lintas produk sehingga perbedaan level antar produk ikut dihitung sebagai variasi yang dijelaskan.
- Hanya satu fitur (`t`), jadi musiman (Lebaran, Nataru) belum tertangkap.

## 6. Langkah lanjutan
- Verifikasi data 2026 (apakah lengkap).
- Analisis Pareto produk per tahun dan kontribusi kategori (agregasi pandas).
- Perbandingan tahun ke tahun (selisih, PnL, growth).
- Tambah fitur musiman (bulan dalam tahun, event Lebaran/Nataru) bila penurunan 2026 terbukti bukan masalah data.
