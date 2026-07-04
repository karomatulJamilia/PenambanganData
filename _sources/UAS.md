# UAS

Nama : Karomatul Jamilia
NIM  : 240411100116
Kelas: Penambangan Data B

# Analisis Higher Education Students Performance Evaluation Menggunakan KNIME

## 1. Latar Belakang

Prestasi akademik mahasiswa dipengaruhi oleh berbagai faktor seperti kondisi keluarga, kebiasaan belajar, aktivitas sosial, dan kehadiran di kelas.

Perguruan tinggi memerlukan informasi mengenai faktor-faktor tersebut untuk membantu meningkatkan performa akademik mahasiswa.

**Tujuan Analisis**
1. Menganalisis faktor-faktor yang memengaruhi nilai akhir mahasiswa.
2. Membuat model prediksi nilai akhir mahasiswa.
3. Menentukan variabel yang paling berpengaruh terhadap performa akademik.

**Pertanyaan**
* Faktor apa yang paling memengaruhi nilai mahasiswa?
* Apakah kebiasaan belajar berpengaruh terhadap nilai akhir?
* Seberapa baik model machine learning memprediksi nilai mahasiswa?

## 2. Sumber Dataset
Dataset berasal dari https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation  .
Informasi dataset:
| Informasi       | Nilai          |
| --------------- | -------------- |
| Jumlah Data     | 145            |
| Jumlah Variabel | 33             |
| Missing Value   | Tidak Ada      |
| Target          | GRADE          |
| Jenis Masalah   | Classification |


**Deskripsi Variabel**
1. Variabel Target
   GRADE
   Kategori:
   * 0 = Fail
   * 1 = DD
   * 2 = DC
   * 3 = CC
   * 4 = CB
   * 5 = BB
   * 6 = BA
   * 7 = AA
  
2. Variabel Prediktor
   * Data demografi mahasiswa
   * Data keluarga
   * Aktivitas akademik
   * Kebiasaan belajar
   * Nilai ujian tengah semester

## 3. Tools yang Digunakan
Analisis dilakukan menggunakan KNIME Analytics Platform.

KNIME dipilih karena menyediakan workflow visual sehingga proses analisis data dan machine learning dapat dilakukan tanpa banyak penulisan kode (low-code).

## 4. Metode yang Digunakan

Pada penelitian ini digunakan tiga algoritma klasifikasi, yaitu **K-Nearest Neighbor (KNN)**, **Decision Tree**, dan **Random Forest**. Ketiga algoritma tersebut dipilih untuk membandingkan performa dalam memprediksi nilai akhir mahasiswa (GRADE).

### 4.1 K-Nearest Neighbor (KNN)

K-Nearest Neighbor (KNN) merupakan algoritma klasifikasi yang menentukan kelas suatu data berdasarkan mayoritas kelas dari sejumlah tetangga terdekat.

**Alasan menggunakan KNN:**
- Mudah diimplementasikan.
- Cocok digunakan sebagai model pembanding.
- Memiliki performa yang baik pada dataset berukuran kecil.



### 4.2 Decision Tree

Decision Tree merupakan algoritma klasifikasi yang membentuk struktur pohon keputusan berdasarkan atribut yang paling baik dalam memisahkan data.

**Alasan menggunakan Decision Tree:**
- Mudah dipahami karena menghasilkan aturan keputusan.
- Dapat menangani data kategorikal dengan baik.
- Mampu menunjukkan proses pengambilan keputusan secara visual.



### 4.3 Random Forest

Random Forest merupakan metode ensemble yang terdiri dari banyak Decision Tree. Hasil prediksi diperoleh berdasarkan voting dari seluruh pohon keputusan yang dibangun.

**Alasan menggunakan Random Forest:**
- Mengurangi risiko overfitting.
- Memiliki performa yang lebih stabil dibandingkan Decision Tree.
- Umumnya menghasilkan akurasi yang lebih baik.



# 5. Workflow Analisis di KNIME

Proses analisis dilakukan menggunakan beberapa node pada KNIME sebagai berikut.

## 5.1 Import Dataset

Node yang digunakan:
- CSV Reader

Fungsi:
- Mengimpor dataset Higher Education Students Performance Evaluation ke dalam workflow KNIME.


## 5.2 Data Preparation

Node yang digunakan:
- Column Filter
- Number to String

Fungsi:

**Column Filter**
- Menghapus atribut yang tidak digunakan pada proses klasifikasi, yaitu:
  - STUDENT ID
  - COURSE ID

**Number to String**
- Mengubah atribut **GRADE** menjadi tipe String agar dikenali sebagai class label oleh algoritma klasifikasi.



## 5.3 Data Partitioning

Node yang digunakan:
- Table Partitioner

Fungsi:
- Membagi dataset menjadi:
  - 80% Data Training
  - 20% Data Testing

Pengaturan yang digunakan:
- Relative Partition
- Stratified Sampling berdasarkan kolom **GRADE**



## 5.4 Pembangunan Model

Model klasifikasi dibangun menggunakan tiga algoritma.

### a. K-Nearest Neighbor
![original image](https://cdn.mathpix.com/snip/images/U6fCgnJtKP0zOQx8hkUf0l9KB1BfgNQDkm9BQQOMZVc.original.fullsize.png)


Node:
- K Nearest Neighbor
- K Nearest Neighbor Predictor


### b. Decision Tree
![original image](https://cdn.mathpix.com/snip/images/cXzmzU5Hk_oAMtgpq3pr-_D1lRyVwXaEMANMbsR8e40.original.fullsize.png)


Node:
- Decision Tree Learner
- Decision Tree Predictor

### c. Random Forest
![original image](https://cdn.mathpix.com/snip/images/7rDKoc0EXGDfTW455IcmtgSNsTlo6mDnvtVKYIYtW8w.original.fullsize.png)


Node:
- Random Forest Learner
- Random Forest Predictor

## 5.5 Evaluasi Model

Node yang digunakan:
- Scorer

Metrik evaluasi yang digunakan:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
### KNN
![original image](https://cdn.mathpix.com/snip/images/L0IZfFPR1oK7VE51dEEDU1jaNAvHnlmFG1-qxygxWfw.original.fullsize.png)

### Decision Tree
![original image](https://cdn.mathpix.com/snip/images/7O1cWSgaFgn38FBLTwiwYbBznENKzd3pOf0RjnU5vUo.original.fullsize.png)

### Random Forest
![original image](https://cdn.mathpix.com/snip/images/lAs4r_tPPPK3IEZvfC0NtG-afF59ASwddd4TQGh9Xtw.original.fullsize.png)

# 6. Hasil Pengujian

Hasil evaluasi ketiga algoritma ditunjukkan pada tabel berikut.

| Metode | Accuracy |
|---------|---------:|
| K-Nearest Neighbor | 22.7% |
| Decision Tree | 24.1% |
| Random Forest | **27.6%** |

Berdasarkan hasil pengujian, algoritma **Random Forest** memperoleh nilai akurasi tertinggi dibandingkan algoritma lainnya.



# 7. Kesimpulan

Berdasarkan hasil analisis menggunakan KNIME, dapat disimpulkan bahwa ketiga algoritma mampu melakukan klasifikasi terhadap nilai akhir mahasiswa, namun memiliki performa yang berbeda.

Hasil pengujian menunjukkan bahwa **Random Forest** memberikan performa terbaik dengan nilai accuracy sebesar **27.6%**, diikuti oleh **Decision Tree** sebesar **24.1%**, dan **K-Nearest Neighbor (KNN)** sebesar **22.7%**.

Dengan demikian, Random Forest menjadi algoritma yang paling sesuai digunakan pada dataset **Higher Education Students Performance Evaluation** dalam penelitian ini.

Meskipun tingkat akurasi yang diperoleh masih relatif rendah, penelitian ini berhasil menunjukkan perbandingan performa ketiga algoritma klasifikasi pada dataset yang digunakan. Hasil ini juga menunjukkan bahwa karakteristik dataset, jumlah data yang terbatas, serta banyaknya kategori pada variabel target memengaruhi tingkat keberhasilan model dalam melakukan prediksi.
