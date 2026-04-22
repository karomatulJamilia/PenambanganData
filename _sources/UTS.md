# UTS Penambangan Data

## 1. Deskripsi Dataset

Dataset yang digunakan merupakan dataset kesuburan tanah yang terdiri dari:

- Jumlah data: 2000 baris
- Jumlah fitur: 10 fitur
- Jenis fitur:
    - 9 fitur numerik
    - 1 fitur kategorikal
- Target/label:
    - Subur
    - Tidak Subur

Dataset ini juga memiliki beberapa nilai yang hilang (missing value) sehingga perlu dilakukan tahap preprocessing sebelum digunakan dalam proses klasifikasi.

![Dataset Awal](https://cdn.mathpix.com/snip/images/PeTq8rmIHsKAzflogb3O9JG0f2l2oq4Sa8eohpfaauY.original.fullsize.png)

---

## 2. Preprocessing Data

### 2.1 Penanganan Missing Value

Missing value diatasi menggunakan metode imputasi:

- Numerik → Mean
- Kategorikal → Modus

Menggunakan Node Missing Value pada KNIME.

![Missing Value](https://cdn.mathpix.com/snip/images/Szdz9o4ZdpKhYHUCC01wu_deLSmuxAobEdAWseH5L3c.original.fullsize.png)

Node Missing Value digunakan untuk mengisi data yang kosong agar dataset tidak mengandung nilai null.

Setelah proses ini, seluruh data yang kosong telah terisi sehingga dataset menjadi lengkap dan siap digunakan.

---

### 2.2 Penghapusan Data Duplikat

Data duplikat dihapus menggunakan Duplicate Row Filter.

**Tujuan:**

- Menghindari bias
- Menghindari data berulang
---

### 2.3 Transformasi Data Kategorikal

Data kategorikal diubah menjadi numerik menggunakan One to Many (One-Hot Encoding).

Contoh:

- Liat → [1 0 0]
- Pasir → [0 1 0]

![One to Many](https://cdn.mathpix.com/snip/images/hVJ5Y46pQVAW9W9Mlhp7xbb8MFPrcr6YxJL6EvxRDgs.original.fullsize.png)

Node Duplicate Row Filter digunakan untuk menghapus data yang memiliki nilai sama agar tidak terjadi bias.
Data kategori diubah menjadi numerik agar bisa diproses oleh KNN.

---

### 2.4 Seleksi Fitur

Menggunakan Column Filter untuk menghapus kolom yang tidak diperlukan.

![original image](https://cdn.mathpix.com/snip/images/Aw6WyYLYTjeHYtZNrc-q76KwNZ9WGdY5upy2FGAD3XE.original.fullsize.png)

**Tujuan:**

- Mengurangi noise
- Mempercepat komputasi

---

### 2.5 Normalisasi Data

Normalisasi dilakukan menggunakan Min-Max Normalization.

**Rumus:**

$$
X' = \frac{X - X_{min}}{X_{max} - X_{min}}
$$

**Penjelasan:**

- X = nilai asli
- X' = hasil normalisasi

![original image](https://cdn.mathpix.com/snip/images/p1yRoc_UZYLs4VP0h905PnMVyAVI8qTpSrUqvrIFoQQ.original.fullsize.png)

---

## 3. Pembagian Data

Data dibagi menggunakan Table Partitioner:

- Training: 80%
- Testing: 20%
- Sampling: Random

![original image](https://cdn.mathpix.com/snip/images/kuMXlczWUfpDy-8CEpDhFjiQLq_0nSnAvwqT8KrMhhM.original.fullsize.png)

Training digunakan untuk melatih model, testing untuk evaluasi.

---

## 4. Metode KNN

KNN (K-Nearest Neighbor) adalah metode klasifikasi berbasis jarak.

**Rumus Euclidean Distance:**

$$
d(x,y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}
$$

**Penjelasan:**

- x = data uji
- y = data training
- n = jumlah fitur

**Langkah KNN:**

1. Hitung jarak ke semua data
2. Urutkan jarak
3. Ambil K terdekat
4. Tentukan kelas mayoritas

Parameter:

- K = 3
---

## 5. Hasil Pengujian

### 5.1 Metrik Evaluasi

**Accuracy:**

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

**Precision:**

$$
Precision = \frac{TP}{TP + FP}
$$

**Recall:**

$$
Recall = \frac{TP}{TP + FN}
$$

**F1-Score:**

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

**Hasil:**

- Accuracy = 1.00
- Precision = 1.00
- Recall = 1.00
- F1-Score = 1.00

![original image](https://cdn.mathpix.com/snip/images/pIzgihHoj77EtOQEUEv-4bYOPWy1PCticOYpefmCLX8.original.fullsize.png)

Model mampu mengklasifikasikan data dengan sangat baik.

---

## 6. Analisis Hasil

Hasil menunjukkan performa sangat tinggi.

Kemungkinan:

- Data memiliki pola jelas
- Model overfitting
- Nilai K kecil

---

## 7. Kesimpulan

1. KNN berhasil digunakan
2. Preprocessing penting
3. Model sangat akurat
4. Kemungkinan overfitting

---

## 8. Lampiran

![original image](https://cdn.mathpix.com/snip/images/N1LO8BGdnZmMjAVUvwUYRB-hWe2YbiuXPEC-mBbVjzc.original.fullsize.png)

