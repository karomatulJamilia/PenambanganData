# Random Forest
## Analisis Data Menggunakan Random Forest

Pada tugas ini dilakukan analisis data menggunakan metode Decision Tree dan Random Forest pada dataset diabetes. Tujuannya adalah membandingkan performa kedua algoritma dalam melakukan klasifikasi data diabetes berdasarkan nilai akurasi yang dihasilkan.

## 1. Penjelasan Algoritma
### 1.1 Decision Tree
Decision Tree adalah algoritma machine learning berbentuk pohon keputusan yang digunakan untuk klasifikasi maupun prediksi.
Algoritma ini bekerja dengan membagi data berdasarkan atribut tertentu hingga menghasilkan keputusan akhir.
### 1.2 Random Forest
Random Forest merupakan pengembangan dari Decision Tree yang menggunakan banyak pohon keputusan (tree).
Hasil prediksi diperoleh dari voting seluruh tree sehingga hasil lebih stabil dan akurat.

**Cara kerja Random Forest**
1. Membuat banyak Decision Tree
2. Setiap tree menggunakan data acak
3. Semua tree melakukan prediksi
4. Hasil voting menentukan prediksi akhir


## 2. Dataset
Dataset yang digunakan pada penelitian ini adalah dataset Pima Indians Diabetes Database yang diperoleh dari Kaggle. Dataset ini berasal dari National Institute of Diabetes and Digestive and Kidney Diseases dan digunakan untuk memprediksi apakah seorang pasien menderita diabetes atau tidak berdasarkan beberapa pengukuran medis. Dataset ini sangat populer dalam bidang Machine Learning karena sering digunakan untuk klasifikasi data kesehatan dan pengujian algoritma machine learning.

Dataset ini berisi data pasien perempuan keturunan Pima Indian yang berusia minimal 21 tahun. Tujuan utama dari dataset ini adalah melakukan prediksi diabetes menggunakan beberapa atribut kesehatan seperti kadar glukosa, tekanan darah, BMI, insulin, dan usia pasien.

https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database


## 3. Workflow KNIME
### 3.1 Workflow Decision Tree

![original image](https://cdn.mathpix.com/snip/images/NmHIsJZ4amDo9KlS3Owr8JejYYL5tfW0Ak-oNZEkdFI.original.fullsize.png)


* Node CSV Reader digunakan untuk membaca dataset diabetes yang berbentuk file .csv.
* Node Missing Value digunakan untuk menangani data kosong atau missing value pada dataset.
* Node Number to String digunakan untuk mengubah tipe data numerik menjadi string. Pada workflow Decision Tree kolom target Outcome diubah menjadi string agar lebih mudah diproses sebagai label klasifikasi.
* Node Table Partitioner digunakan untuk membagi dataset menjadi data training dan data testing (80% training dan 20% testing).
* Node Decision Tree Learner digunakan untuk membuat model Decision Tree berdasarkan data training menggunakan gain ratio.
* Node Decision Tree Predictor digunakan untuk melakukan prediksi terhadap data testing menggunakan model yang telah dibuat.
* Node Scorer digunakan untuk mengevaluasi hasil prediksi model.
### 3.2 Workflow Random Forest dengan Python Script

![original image](https://cdn.mathpix.com/snip/images/9mGWzxwGErt6UvK0ZKvxXcx6LDxr-Bg1IEolafn43NQ.original.fullsize.png)

#### 3.2.1 Python Script 1 (Training dan Save Model)

**Script Python 1 :**
```
from sklearn.ensemble import RandomForestClassifier
import joblib
import knime.scripting.io as knio

# Ambil data train dari KNIME
train_df = knio.input_tables[0].to_pandas()

# Pisahkan fitur dan target
X_train = train_df.drop(columns=["Outcome"])
y_train = train_df["Outcome"]

# Buat model Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Training model
print("Training model...")
model.fit(X_train, y_train)

# Simpan model ke file .pkl
joblib.dump(model, "random_forest_diabetes.pkl")

print("Model berhasil disimpan!")

# Output ke node berikutnya
knio.output_tables[0] = knio.Table.from_pandas(train_df)
```

Python Script 1 berfungsi sebagai proses training model Random Forest menggunakan data training yang berasal dari node Table Partitioner. Pada script ini data dipisahkan menjadi fitur dan target (Outcome), kemudian model Random Forest dibuat menggunakan library scikit-learn dengan jumlah 100 decision tree. Setelah model selesai dilatih menggunakan data training, model tersebut disimpan ke dalam file berformat .pkl menggunakan library joblib. File .pkl ini nantinya digunakan kembali pada proses testing di Python Script 2. Dengan demikian Python Script 1 memiliki fungsi utama sebagai proses pembuatan, pelatihan, dan penyimpanan model machine learning.

#### 3.2.2 Python Script 2 (Load Model dan Testing)
**Script Python 2 :**
```
from sklearn.metrics import accuracy_score
import joblib
import knime.scripting.io as knio

# Input dari KNIME
train_df = knio.input_tables[0].to_pandas()
test_df = knio.input_tables[1].to_pandas()

# Pisahkan fitur dan target test
X_test = test_df.drop(columns=["Outcome"])
y_test = test_df["Outcome"]

# Load model
model = joblib.load("random_forest_diabetes.pkl")

# Prediksi data test
y_pred = model.predict(X_test)

# Hitung accuracy
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)

# Tambahkan hasil prediksi
output_df = test_df.copy()
output_df["Prediction"] = y_pred

# Kirim hasil ke KNIME
knio.output_tables[0] = knio.Table.from_pandas(output_df)
```

Python Script 2 berfungsi sebagai proses testing dan prediksi data menggunakan model Random Forest yang telah disimpan sebelumnya. Pada script ini model .pkl dibaca kembali menggunakan joblib.load(), kemudian data testing dari Table Partitioner digunakan untuk melakukan prediksi klasifikasi diabetes. Hasil prediksi dibandingkan dengan data asli (Outcome) untuk menghitung nilai accuracy menggunakan accuracy_score. Selain itu script ini juga menambahkan kolom Prediction ke dalam output data sehingga hasil prediksi dapat ditampilkan dan dievaluasi pada node Scorer. Dengan demikian Python Script 2 memiliki fungsi utama sebagai proses membaca model, melakukan prediksi data testing, dan menghitung performa model Random Forest.

## 4. Perbandingan Penggunaan Decision Tree dan Random Forest

**Scorer Decision Tree**
![original image](https://cdn.mathpix.com/snip/images/voZ8XQgj1xJhVQxzZLIvGjmI0tMtuqPmEyoVVqtRkQ4.original.fullsize.png)

Berdasarkan hasil node Scorer, metode Decision Tree menghasilkan nilai accuracy sebesar 0.708 .

Hasil confusion matrix menunjukkan bahwa model mampu melakukan klasifikasi data diabetes dengan cukup baik, namun masih terdapat beberapa kesalahan prediksi antara kelas diabetes dan non-diabetes.

Pada hasil tersebut:
* sebanyak 83 data berhasil diprediksi benar sebagai non-diabetes
* sebanyak 25 data berhasil diprediksi benar sebagai diabetes
* sedangkan beberapa data masih mengalami kesalahan klasifikasi

Nilai precision, recall, dan F-measure pada Decision Tree menunjukkan performa yang cukup baik, namun model masih cenderung mengalami overfitting karena hanya menggunakan satu pohon keputusan. Hal ini menyebabkan hasil prediksi menjadi kurang stabil ketika diuji menggunakan data testing.

**Scorer Random Forest**
![original image](https://cdn.mathpix.com/snip/images/H_n_a_L36LGJL3uHlDIwb1BotT2ltZbW-sB7OR6btC8.original.fullsize.png)

Berdasarkan hasil node Scorer, metode Random Forest menghasilkan nilai accuracy sebesar 0.747.

Hasil ini menunjukkan bahwa Random Forest memiliki performa yang lebih baik dibandingkan Decision Tree pada dataset diabetes.

Pada confusion matrix:
* sebanyak 84 data berhasil diprediksi benar sebagai non-diabetes
* sebanyak 25 data berhasil diprediksi benar sebagai diabetes
* jumlah kesalahan prediksi lebih kecil dibandingkan Decision Tree

Hal ini terjadi karena Random Forest menggunakan banyak Decision Tree dalam proses klasifikasi sehingga hasil prediksi menjadi lebih stabil dan mampu mengurangi overfitting.

Selain itu nilai precision, recall, dan F-measure pada Random Forest juga menunjukkan hasil yang lebih baik dibandingkan Decision Tree, terutama dalam membedakan pasien diabetes dan non-diabetes.

| Metode        | Accuracy |
| ------------- | -------- |
| Decision Tree | 70.08%   |
| Random Forest | 74.04%   |



| Aspek        | Decision Tree | Random Forest  |
| ------------ | ------------- | -------------- |
| Jumlah Tree  | 1 tree        | Banyak tree    |
| Akurasi      | Lebih rendah  | Lebih tinggi   |
| Overfitting  | Mudah terjadi | Lebih kecil    |
| Kecepatan    | Lebih cepat   | Lebih lambat   |
| Stabilitas   | Kurang stabil | Lebih stabil   |
| Kompleksitas | Sederhana     | Lebih kompleks |

Berdasarkan hasil pengujian dapat disimpulkan bahwa metode Random Forest menghasilkan accuracy lebih tinggi dibandingkan Decision Tree. Selisih accuracy memang tidak terlalu besar, namun Random Forest menghasilkan prediksi yang lebih stabil karena menggunakan kombinasi banyak pohon keputusan (tree). Oleh karena itu Random Forest lebih baik digunakan untuk proses klasifikasi dataset diabetes dibandingkan Decision Tree tunggal.