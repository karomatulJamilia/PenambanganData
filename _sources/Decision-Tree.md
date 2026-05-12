# Decision Tree 
## Analisis Data Menggunakan Algoritma Decision Tree dengan Metode Gain Ratio pada KNIME


Tugas ini bertujuan untuk melakukan analisis data menggunakan metode Pohon Keputusan (Decision Tree) untuk membangun model klasifikasi. Model ini digunakan untuk memprediksi atau menentukan kelas dari suatu data, misalnya menentukan apakah suatu mobil layak dibeli atau tidak berdasarkan atribut-atributnya.

## 1. Konsep Dasar Decision Tree

Decision Tree adalah metode klasifikasi yang membentuk struktur seperti pohon, di mana:

* Node (simpul) mewakili atribut (misalnya safety, buying, dll)
* Cabang mewakili nilai dari atribut tersebut
* Daun (leaf) berisi hasil klasifikasi (class)

Model ini bekerja dengan cara membagi data menjadi kelompok-kelompok yang lebih kecil berdasarkan atribut tertentu hingga menghasilkan keputusan akhir.

Contoh aturan:

* Jika safety = low maka class = unacc
* Jika safety = high dan persons = more maka class = vgood

## 2. Konsep Gain Ratio
Dalam membangun Decision Tree, diperlukan metode untuk memilih atribut terbaik sebagai pemisah data. Salah satu metode tersebut adalah Gain Ratio.
**a. Entropy**
Entropy digunakan untuk mengukur tingkat ketidakpastian dalam data.
* Jika data masih bercampur → entropy tinggi
* Jika data sudah seragam → entropy rendah

Rumus Entropy:

$Entropy(S) = - \sum_{i=1}^{n} p_i \log_2 p_i$

Keterangan:

$S$ = dataset
$p_i$ = proporsi kelas ke-i

Rumus Weighted Entropy:

$Entropy(S, A) = \sum_{j=1}^{v} \frac{|S_j|}{|S|} \cdot Entropy(S_j)$

Keterangan:
$A$ = atribut
$S_j$ = subset ke-j
$v$ = jumlah nilai atribut

**b. Information Gain**
Information Gain mengukur seberapa baik suatu atribut dalam memisahkan data menjadi lebih teratur.

Rumus:

$Gain(S, A) = Entropy(S) - Entropy(S, A)$

**c. Split Information**
Split Information mengukur seberapa besar pembagian data yang dilakukan oleh suatu atribut.

Rumus:

$SplitInfo(S, A) = - \sum_{j=1}^{v} \frac{|S_j|}{|S|} \log_2 \left(\frac{|S_j|}{|S|}\right)$

**d. Gain Ratio**
Gain Ratio merupakan perbaikan dari Information Gain, dengan rumus:

$GainRatio(S, A) = \frac{Gain(S, A)}{SplitInfo(S, A)}$

Tujuan Gain Ratio:

* Menghindari pemilihan atribut yang memiliki terlalu banyak nilai
* Memilih atribut terbaik untuk dijadikan root (akar pohon).

Metode Gain Ratio dipilih karena mampu mengatasi kelemahan Information Gain yang cenderung memilih atribut dengan jumlah kategori sangat banyak. Dengan normalisasi menggunakan Split Information, Gain Ratio menghasilkan pemilihan atribut yang lebih adil dan optimal dalam pembentukan pohon keputusan.

## 3. Dataset
### 3.1 Deskripsi Dataset
Dataset Car Evaluation digunakan untuk melakukan klasifikasi terhadap kualitas mobil. Data ini berisi atribut-atribut yang menggambarkan kondisi mobil, seperti harga, biaya perawatan, kapasitas, dan tingkat keamanan.
Dataset Car Evaluation dipilih karena seluruh atribut bersifat kategorikal sehingga sangat cocok digunakan pada algoritma Decision Tree berbasis Gain Ratio. Selain itu, dataset ini umum digunakan dalam studi klasifikasi karena memiliki label kelas yang jelas dan pola atribut yang dapat divisualisasikan dalam bentuk pohon keputusan.


Dataset ini memiliki karakteristik sebagai berikut:
* Jumlah atribut: 6 atribut input dan 1 atribut target
* Tipe data: seluruh atribut berupa kategori (non-numerik)
* Digunakan untuk: klasifikasi

Sumber dataset : https://www.kaggle.com/datasets/elikplim/car-evaluation-data-set

## 4. Implementasi Decision Tree di KNIME

Pada tahap ini dilakukan implementasi model klasifikasi menggunakan software KNIME. Proses dimulai dari pembacaan dataset, pembagian data, pembuatan model Decision Tree menggunakan kriteria Gain Ratio, hingga evaluasi hasil model.
### 4.1 Tujuan Penggunaan KNIME
Dalam tugas ini, KNIME digunakan untuk:
* Mengolah dataset
* Membangun model klasifikasi menggunakan Decision Tree
* Menentukan atribut terbaik dengan metode Gain Ratio
* Mengevaluasi performa model
### 4.2  Alur Node yang Digunakan


Alur workflow dalam KNIME adalah sebagai berikut:
Excel Reader → Table Partitioner → Decision Tree Learner → Decision Tree Predictor → Scorer

![original image](https://cdn.mathpix.com/snip/images/i_y1M4_wmu6WBywIMwvB69nprHj5d1MJBEhXA8w1tkU.original.fullsize.png)

1. Excel Reader
   Membaca dataset dari file Excel/CSV
2. Table Patitioner
   Node Partitioning digunakan untuk membagi dataset menjadi data training dan data testing. Pembagian ini bertujuan agar model dapat dilatih dan diuji untuk mengetahui performanya.
   Fungsi:
   * Membagi data menjadi:
     - data training (70%)
     - data testing (30%)

   Tujuan:
   * Agar model tidak hanya menghafal data (overfitting)

4. Decision Tree Learner
   Pada node Decision Tree Learner di KNIME, Gain Ratio digunakan sebagai kriteria untuk menentukan atribut terbaik dalam membangun pohon keputusan.
   Dengan menggunakan Gain Ratio:
   * Model akan memilih atribut yang paling efektif dalam membagi data
   * Pohon yang dihasilkan menjadi lebih optimal dan tidak bias.
     ![original image](https://cdn.mathpix.com/snip/images/gg-E3oiqiFfNCiFfCKSiBb9BKOgYKTsgDYMDaC6ztwc.original.fullsize.png)


5. Decision Tree Predictor
   Menggunakan model untuk memprediksi data testing
   ![original image](https://cdn.mathpix.com/snip/images/7dfzrzNwnO3peDq_Q5OwYVuGfaa7mTg4ct1SKCXokfQ.original.fullsize.png)

   Berdasarkan hasil pembentukan **Decision Tree** pada KNIME, atribut **safety** dipilih sebagai *root node* (akar pohon). Hal ini menunjukkan bahwa atribut keselamatan memiliki pengaruh paling besar dalam menentukan kualitas mobil berdasarkan perhitungan **Gain Ratio**.

Dari hasil pohon keputusan terlihat bahwa:

- Jika **safety = low**, maka seluruh data diklasifikasikan sebagai **unacc (unacceptable)** dengan tingkat kemurnian data sebesar **100%**. Hal ini menunjukkan bahwa mobil dengan tingkat keamanan rendah dianggap tidak layak.

- Jika **safety = high**, maka data terbagi ke beberapa kelas seperti **unacc**, **acc**, **good**, dan **vgood**, sehingga diperlukan percabangan lanjutan untuk menentukan klasifikasi akhir secara lebih spesifik.

- Jika **safety = med**, mayoritas data masih berada pada kelas **unacc**, namun terdapat sebagian data yang termasuk ke dalam kelas **acc** dan **good**, sehingga model tetap melakukan proses pembagian lebih lanjut.

Berdasarkan hasil tersebut dapat disimpulkan bahwa faktor **keamanan (safety)** menjadi atribut paling dominan dalam menentukan tingkat kelayakan mobil pada dataset **Car Evaluation**.


7. Scorer
   Node Scorer digunakan untuk mengevaluasi performa model dengan melihat nilai accuracy, precision, recall, dan f-measure.
   ![original image](https://cdn.mathpix.com/snip/images/v52jTOOZWZVxDIP-fAr7MhuIc3GT9dzJZp-0p6rxtH4.original.fullsize.png)
Berdasarkan hasil pengujian, diperoleh nilai accuracy sebesar 0.919 atau 91.9%. Nilai accuracy sebesar 91.9% menunjukkan bahwa model memiliki performa klasifikasi yang sangat baik. Hal ini berarti sebagian besar data testing berhasil diprediksi dengan benar oleh model Decision Tree menggunakan metode Gain Ratio. Hal ini menunjukkan bahwa model mampu mengklasifikasikan data dengan tingkat ketepatan yang tinggi.
* Accuracy : Nilai accuracy sebesar 91.9% menunjukkan bahwa sebagian besar data berhasil diklasifikasikan dengan benar.
* Precision :Precision menunjukkan tingkat ketepatan prediksi untuk setiap kelas. Nilai precision yang tinggi menandakan bahwa model jarang melakukan kesalahan dalam memprediksi suatu kelas tertentu.
* Recall (Sensitivity) : Recall menunjukkan kemampuan model dalam menemukan semua data yang termasuk dalam suatu kelas. Nilai recall yang baik menunjukkan bahwa model mampu mengenali sebagian besar data dengan benar.
* F-Measure : -measure merupakan kombinasi antara precision dan recall. Nilai ini digunakan untuk menyeimbangkan antara ketepatan dan kelengkapan prediksi model.
* Cohen’s Kappa : Cohen’s Kappa digunakan untuk mengukur tingkat kesepakatan antara hasil prediksi model dan data sebenarnya. Nilai sebesar 0.83 menunjukkan bahwa model memiliki tingkat performa yang sangat baik.