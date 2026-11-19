# Analisis Kinerja Penjualan Marketplace Online Berbasis Machine Learning

Proyek analisis dan peramalan penjualan menggunakan Python dengan menerapkan metodologi **CRISP-DM (Cross-Industry Standard Process for Data Mining)**.

## 1. Gambaran Umum Proyek

Proyek ini bertujuan untuk menganalisis data historis penjualan dari sebuah marketplace online serta menerapkan berbagai metode machine learning dan statistik untuk mengevaluasi kinerja penjualan dan memprediksi penjualan di masa mendatang.

Tujuan utama proyek:

- Menganalisis pola penjualan historis dan kinerja setiap produk.
- Melakukan pembersihan dan persiapan data transaksi marketplace.
- Membandingkan 12 model peramalan menggunakan validasi berdasarkan urutan waktu.
- Mengidentifikasi strategi peramalan dengan performa terbaik.
- Mengevaluasi hasil prediksi menggunakan data pengujian yang belum pernah digunakan dalam pelatihan.
- Menghasilkan prediksi penjualan untuk setiap produk.

**Periode data:** Januari 2024 hingga Desember 2025  
**Periode peramalan:** Januari hingga Maret 2026  
**Jumlah produk:** 20 produk  
**Dataset hasil pembersihan:** 41.954 baris  
**Dataset peramalan bulanan:** 480 observasi produk-bulan

## 2. Teknologi yang Digunakan

| Teknologi    | Kegunaan                                      |
| ------------ | --------------------------------------------- |
| Python       | Bahasa pemrograman utama                      |
| Pandas       | Manipulasi dan pemrosesan data                |
| NumPy        | Komputasi numerik                             |
| Scikit-learn | Algoritma machine learning dan evaluasi model |
| Matplotlib   | Visualisasi data                              |
| CSV          | Penyimpanan dataset dan hasil eksperimen      |

## 3. Metodologi Penelitian: CRISP-DM

Proyek ini mengikuti enam tahapan metodologi CRISP-DM.

### 3.1 Business Understanding (Pemahaman Bisnis)

Tahap ini berfokus pada penentuan tujuan analisis kinerja penjualan marketplace serta pengembangan model peramalan untuk mendukung perencanaan dan pengambilan keputusan bisnis.

### 3.2 Data Understanding (Pemahaman Data)

Dataset awal diperoleh dari Kaggle dan terdiri dari empat file utama:

| Dataset         | Deskripsi                       |
| --------------- | ------------------------------- |
| `orders.csv`    | Informasi transaksi dan pesanan |
| `customers.csv` | Informasi pelanggan             |
| `payments.csv`  | Informasi pembayaran            |
| `products.csv`  | Informasi produk                |

Analisis menggunakan data dari tahun kalender 2024 dan 2025. Data historis tahun 2026 yang belum lengkap tidak disertakan dalam analisis.

### 3.3 Data Preparation (Persiapan Data)

Tahapan persiapan data meliputi:

1. Membaca seluruh dataset awal.
2. Mengidentifikasi nilai kosong, duplikasi, dan data yang tidak valid.
3. Menghapus data pesanan duplikat.
4. Menghapus baris dengan tanggal, kuantitas, atau diskon yang tidak valid.
5. Mengisi nilai usia pelanggan yang kosong menggunakan median.
6. Mengisi informasi kota dan metode pembayaran yang kosong menggunakan nilai `Unknown`.
7. Menyeragamkan penulisan nama kota.
8. Menghapus data pelanggan yang tidak memiliki referensi yang valid.
9. Menggabungkan dataset berdasarkan relasi antar tabel yang telah divalidasi.
10. Menghitung nilai penjualan setiap transaksi.

Rumus perhitungan penjualan:

```python
Sales = Quantity * UnitPrice * (1 - Discount)
```

Hasil akhir proses pembersihan menghasilkan **41.954 baris dan 17 kolom**.

Selanjutnya, data transaksi diagregasikan menjadi data penjualan bulanan untuk setiap produk.

Hasil agregasi terdiri dari:

- 20 produk.
- 24 bulan.
- 480 observasi produk-bulan.

Kombinasi produk dan bulan yang tidak memiliki transaksi direpresentasikan sebagai nilai penjualan nol.

Hanya transaksi dengan status `Completed` yang diperhitungkan dalam perhitungan penjualan.

### 3.4 Modeling (Pemodelan)

Target peramalan adalah nilai penjualan bulanan (`Sales`).

Fitur waktu `t` digunakan untuk merepresentasikan indeks bulan secara berurutan.

Sebanyak 12 model dan baseline peramalan diuji:

| No. | Model                 | Konfigurasi              |
| --: | --------------------- | ------------------------ |
|   1 | Naive                 | Rata-rata data pelatihan |
|   2 | Linear Regression     | Model linear             |
|   3 | Polynomial Regression | Derajat 2                |
|   4 | Polynomial Regression | Derajat 5                |
|   5 | Ridge Regression      | Alpha = 0.1              |
|   6 | Ridge Regression      | Alpha = 1                |
|   7 | Ridge Regression      | Alpha = 10               |
|   8 | Lasso Regression      | Alpha = 0.1              |
|   9 | Lasso Regression      | Alpha = 1                |
|  10 | Lasso Regression      | Alpha = 10               |
|  11 | Seasonal Naive        | Baseline musiman         |
|  12 | Moving Average        | Jendela 3 bulan          |

Model regresi dievaluasi secara terpisah untuk setiap produk.

#### Pembagian Data Berdasarkan Urutan Waktu

Pembagian dataset dilakukan secara kronologis sebagai berikut:

| Dataset    | Periode             |   Durasi |
| ---------- | ------------------- | -------: |
| Training   | Jan 2024 - Mei 2025 | 17 bulan |
| Validation | Jun 2025 - Sep 2025 |  4 bulan |
| Testing    | Okt 2025 - Des 2025 |  3 bulan |

Pembagian data secara acak tidak digunakan karena urutan waktu harus dipertahankan dalam analisis deret waktu (_time series_).

#### Walk-Forward Validation

Evaluasi model menggunakan pendekatan _expanding window_ dengan tiga fold:

| Fold | Periode Training  | Periode Validation |
| ---- | ----------------- | ------------------ |
| 1    | Jan-Des 2024      | Jan-Mar 2025       |
| 2    | Jan 2024-Mar 2025 | Apr-Jun 2025       |
| 3    | Jan 2024-Jun 2025 | Jul-Sep 2025       |

Setiap model dievaluasi pada ketiga fold tersebut.

Model dengan nilai rata-rata RMSE terendah dipilih sebagai model dengan performa terbaik.

Untuk pemilihan model per produk, model dengan rata-rata RMSE terendah dari ketiga fold dipilih secara individual untuk setiap produk.

### 3.5 Evaluation (Evaluasi)

Kinerja model dievaluasi menggunakan beberapa metrik:

| Metrik | Kegunaan                                                |
| ------ | ------------------------------------------------------- |
| MAE    | Mengukur rata-rata kesalahan absolut prediksi           |
| MSE    | Mengukur rata-rata kuadrat kesalahan prediksi           |
| RMSE   | Mengukur akar rata-rata kuadrat kesalahan prediksi      |
| WAPE   | Mengukur persentase kesalahan absolut berbobot          |
| R²     | Mengukur kemampuan model dalam menjelaskan variasi data |

Evaluasi akhir membandingkan tiga strategi peramalan:

1. **Global Training Mean Baseline:** Menggunakan rata-rata periode training secara global.
2. **Best Global ML:** Menggunakan Ridge Regression dengan alpha 10.
3. **Per-Product Selection:** Menggunakan model terbaik yang dipilih secara individual untuk setiap produk berdasarkan hasil walk-forward validation.

Model akhir dilatih ulang menggunakan data pengembangan dari Januari 2024 hingga September 2025, kemudian dievaluasi menggunakan data pengujian yang tidak digunakan dalam proses pemilihan model, yaitu Oktober hingga Desember 2025.

#### Hasil Evaluasi Akhir

| Strategi              |        MAE |         RMSE |       WAPE |        R² |
| --------------------- | ---------: | -----------: | ---------: | --------: |
| Global Training Mean Baseline | **769.89** | **1,101.98** | **13.59%** | **0.858** |
| Best Global ML        |     818.86 |     1,173.88 |     14.46% |     0.839 |
| Per-Product Selection |     928.47 |     1,326.27 |     16.39% |     0.794 |

**Temuan utama:** Strategi Global Training Mean Baseline memperoleh performa agregat terbaik pada dataset pengujian akhir dibandingkan Ridge Regression maupun strategi pemilihan model secara individual per produk.

### 3.6 Deployment (Penerapan)

Model peramalan terpilih digunakan untuk menghasilkan 60 prediksi yang mencakup 20 produk selama tiga bulan mendatang:

- Januari 2026.
- Februari 2026.
- Maret 2026.

Hasil prediksi disimpan dalam file `temp/future_forecast.csv`.

Prediksi tersebut dihasilkan berdasarkan data historis yang tersedia hingga Desember 2025. Hasil ini merupakan estimasi dari model, bukan data penjualan aktual.

## 4. Instalasi dan Persiapan

### Persyaratan

- Python 3.11 atau versi yang kompatibel.
- pip.
- Git (opsional).

### Clone Repository

```bash
git clone <repository-url>
cd sales-performance-analysis
```

### Membuat Virtual Environment

Linux / macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### Instalasi Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Cara Menggunakan Proyek

File utama untuk menjalankan pipeline adalah `main.py`.

Pipeline mendukung tiga mode eksekusi:

### Menjalankan Data Cleaning

```bash
python main.py clean
```

Perintah ini memproses dataset awal dan menghasilkan data bersih yang diperlukan untuk tahap pemodelan.

### Menjalankan Pemodelan dan Evaluasi

```bash
python main.py model
```

Perintah ini menjalankan agregasi penjualan bulanan, pelatihan dan evaluasi model, walk-forward validation, pemilihan model terbaik, serta pembuatan prediksi penjualan.

### Menjalankan Seluruh Pipeline

```bash
python main.py all
```

Perintah ini menjalankan seluruh workflow secara berurutan:

1. Data preparation dan validasi.
2. Training, walk-forward validation, evaluasi, dan forecasting.
3. Pembuatan delapan visualisasi.
4. Analisis insights deterministik berbasis artifact CSV.
5. Pembuatan laporan HTML lokal.

Laporan utama tersedia di `reports/latest/index.html`.

### Perintah pipeline tambahan

```bash
python main.py visualize
python main.py insights
python main.py report
```

Perintah tersebut digunakan untuk menjalankan tahap individual setelah artifact dari tahap sebelumnya tersedia. Semua perintah mendukung `--data-dir`, `--out-dir`, dan `--config`; lokasi chart dan laporan dapat diubah dengan `--visualizations-dir` dan `--report-dir`.

### Menghasilkan Visualisasi

Setelah seluruh file CSV yang diperlukan tersedia, jalankan:

```bash
python main.py visualize
```

Seluruh grafik akan disimpan dalam direktori `visualizations/`. Implementasi modular berada di `salespipe/visualization.py`; `visualize_sales.py` dipertahankan sebagai script analisis lama.

**Catatan:** Script visualisasi membaca hasil CSV yang sudah tersedia. Jalankan pipeline pemodelan terlebih dahulu apabila file hasil pemodelan belum tersedia atau perlu dibuat ulang.

## 6. Struktur Direktori Proyek

```text
sales-performance-analysis/
│
├── datasets/
│   ├── orders.csv
│   ├── customers.csv
│   ├── payments.csv
│   ├── products.csv
│   └── clean_final_data.csv
│
├── salespipe/
│   ├── __init__.py
│   ├── cleaning.py
│   ├── config.py
│   └── modeling.py
│
├── temp/
│   ├── clean_merged.csv
│   ├── monthly.csv
│   ├── train.csv
│   ├── validasi.csv
│   ├── test.csv
│   ├── model_metrics.csv
│   ├── walk_forward_metrics.csv
│   ├── walk_forward_summary.csv
│   ├── final_selected_models.csv
│   ├── final_test_predictions.csv
│   ├── final_test_metrics.csv
│   ├── final_test_error_analysis.csv
│   └── future_forecast.csv
│
├── visualizations/
│   ├── overall_monthly_sales.png
│   ├── product_monthly_sales.png
│   ├── product_sales_comparison.png
│   ├── model_performance_comparison.png
│   ├── actual_vs_predicted_sales.png
│   ├── forecasting_error_contribution.png
│   ├── overall_future_forecast.png
│   └── future_forecast_by_product.png
│
├── main.py
├── predict.py
├── validate_forecast.py
├── visualize_sales.py
│
├── analyze_errors.py
├── analyze_product_metrics.py
├── analyze_product_models.py
│
├── config.example.json
├── requirements.txt
├── README.md
└── .gitignore
```

### Fungsi Komponen Utama

| File / Direktori             | Fungsi                                                              |
| ---------------------------- | ------------------------------------------------------------------- |
| `datasets/`                  | Menyimpan dataset awal dan dataset hasil pembersihan                |
| `salespipe/cleaning.py`      | Membaca, membersihkan, menggabungkan data, dan menghitung penjualan |
| `salespipe/config.py`        | Mengelola konfigurasi pipeline                                      |
| `salespipe/modeling.py`      | Agregasi bulanan, pelatihan model, validasi, dan peramalan          |
| `main.py`                    | Titik masuk utama untuk menjalankan pipeline                        |
| `predict.py`                 | Pembuatan atau pemeriksaan hasil prediksi                           |
| `validate_forecast.py`       | Validasi hasil peramalan                                            |
| `visualize_sales.py`         | Menghasilkan delapan visualisasi                                    |
| `analyze_errors.py`          | Analisis kesalahan prediksi                                         |
| `analyze_product_metrics.py` | Analisis metrik evaluasi per produk                                 |
| `analyze_product_models.py`  | Perbandingan model per produk                                       |
| `temp/`                      | Menyimpan dataset perantara dan hasil eksperimen                    |
| `visualizations/`            | Menyimpan grafik hasil analisis                                     |
| `requirements.txt`           | Daftar dependensi Python                                            |
| `config.example.json`        | Contoh konfigurasi proyek                                           |

Direktori `temp/` berisi hasil eksperimen dan dataset yang dihasilkan oleh pipeline. File CSV di dalamnya digunakan untuk analisis, evaluasi, dan visualisasi.

## 7. Visualisasi yang Dihasilkan

Proyek ini menghasilkan delapan visualisasi utama:

| No. | Visualisasi                      | Deskripsi                                                         |
| --: | -------------------------------- | ----------------------------------------------------------------- |
|   1 | Overall Monthly Sales            | Total penjualan marketplace selama 2024-2025                      |
|   2 | Monthly Sales by Product         | Tren penjualan bulanan setiap produk                              |
|   3 | Total Sales by Product           | Perbandingan total penjualan antarproduk                          |
|   4 | Model Performance Comparison     | Perbandingan rata-rata RMSE dari walk-forward validation          |
|   5 | Actual vs Predicted Sales        | Perbandingan prediksi dengan penjualan aktual pada data pengujian |
|   6 | Forecasting Error Contribution   | Produk dengan kontribusi kesalahan prediksi terbesar              |
|   7 | Overall Future Sales Forecast    | Prediksi agregat penjualan Januari-Maret 2026                     |
|   8 | Future Sales Forecast by Product | Prediksi penjualan setiap produk beserta model terpilih           |

Seluruh visualisasi dibuat menggunakan Matplotlib dan disimpan dalam format PNG dengan resolusi 300 DPI.

## 8. Keterbatasan Penelitian

Beberapa keterbatasan yang perlu diperhatikan dalam menginterpretasikan hasil penelitian:

- Periode data historis hanya mencakup 24 bulan.
- Dataset terdiri dari 20 produk.
- Periode pengujian akhir hanya mencakup tiga bulan.
- Model peramalan terutama menggunakan informasi urutan waktu dan belum memanfaatkan banyak variabel eksternal sebagai faktor penjelas.
- Model belum tentu dapat menangkap perubahan pasar yang tidak terduga, promosi, maupun pola musiman.
- Prediksi masa depan belum diverifikasi menggunakan data penjualan aktual tahun 2026.

Oleh karena itu, hasil penelitian perlu diinterpretasikan sesuai dengan cakupan dataset dan metodologi eksperimen yang digunakan.

## 9. Referensi Dataset

Dataset sumber:

[E-Commerce Sales Data Analysis & EDA](https://www.kaggle.com/datasets/erfan4524/e-commerce-sales-data-analysis-and-eda/data)

Sumber: Kaggle.

## 10. Status Proyek

| Tahapan CRISP-DM       | Status                                     |
| ---------------------- | ------------------------------------------ |
| Business Understanding | Selesai                                    |
| Data Understanding     | Selesai                                    |
| Data Preparation       | Selesai                                    |
| Modeling               | Selesai                                    |
| Evaluation             | Selesai untuk eksperimen saat ini          |
| Deployment             | Pembuatan prediksi dan visualisasi selesai |

---

**Judul Proyek:** Analisis Kinerja Penjualan Marketplace Online Berbasis Machine Learning Menggunakan Kerangka CRISP-DM

**Tujuan:** Penelitian akademik dan penerapan metode peramalan deret waktu untuk menganalisis kinerja penjualan marketplace menggunakan machine learning dan metode statistik sebagai baseline.
