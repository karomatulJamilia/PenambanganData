## 1. Missing Value Imputation dengan WKNN

Dalam proses pengolahan data, sering ditemukan data yang tidak lengkap atau *missing value*. Missing value dapat mempengaruhi hasil analisis karena informasi yang dibutuhkan tidak tersedia secara lengkap. Oleh karena itu diperlukan metode untuk mengisi nilai yang hilang tersebut.

Salah satu metode yang dapat digunakan adalah **Weighted K-Nearest Neighbor Imputation (WKNN)**. Metode ini merupakan pengembangan dari metode K-Nearest Neighbor yang digunakan untuk mengisi nilai yang hilang dengan mempertimbangkan kedekatan antar data serta memberikan bobot pada tetangga terdekat.

Pada metode WKNN, data yang memiliki missing value akan dibandingkan dengan data lain yang lengkap. Kemudian dihitung jarak atau kemiripan antara data tersebut. Nilai yang hilang akan diperkirakan menggunakan nilai dari beberapa data yang paling mirip dengan mempertimbangkan bobot dari masing-masing tetangga.

#### Langkah-langkah metode WKNN

1. Menentukan data yang memiliki missing value.
2. Menghitung jarak atau kemiripan antara data yang memiliki missing value dengan data lainnya.
3. Menentukan beberapa tetangga terdekat (K tetangga).
4. Menghitung bobot berdasarkan nilai kemiripan.
5. Menghitung nilai imputasi menggunakan rata-rata berbobot dari tetangga terdekat.

#### Rumus Similarity pada WKNN

Nilai similarity dihitung menggunakan rumus berikut:

$$
S_i = \frac{1}{\sum (Y_{ih} - Y_{jh})^2}
$$

#### Keterangan simbol

$$
\begin{aligned}
S_i &= \text{nilai similarity antara data } i \text{ dan } j \\
Y_{ih} &= \text{nilai atribut ke-}h \text{ pada data } i \\
Y_{jh} &= \text{nilai atribut ke-}h \text{ pada data } j
\end{aligned}
$$

#### Rumus Perhitungan Nilai Imputasi

Setelah nilai similarity diperoleh, nilai imputasi dihitung menggunakan rata-rata berbobot sebagai berikut:

$$
\hat{y}_{ih} =
\frac{\sum_{j \in IK_{ih}} s_i(y_j) y_{jh}}
{\sum_{j \in IK_{ih}} s_i(y_j)}
$$

### Keterangan simbol

$$
\begin{aligned}
\hat{y}_{ih} &= \text{nilai imputasi untuk data } i \text{ pada atribut } h \\
IK_{ih} &= \text{himpunan tetangga terdekat dari data } i \\
s_i(y_j) &= \text{nilai similarity antara data } i \text{ dan } j \\
y_{jh} &= \text{nilai atribut } h \text{ pada data tetangga } j
\end{aligned}
$$

Metode WKNN memanfaatkan nilai kemiripan antar data untuk memperkirakan nilai yang hilang sehingga nilai yang dihasilkan lebih mendekati kondisi data sebenarnya.

#### Dataset yang digunakan
Pada tugas ini digunakan dataset dari platform Kaggle yaitu Boston Housing Dataset https://www.kaggle.com/datasets/altavish/boston-housing-dataset.

Dataset Boston Housing merupakan dataset yang berisi informasi mengenai berbagai karakteristik wilayah perumahan di kota Boston, Amerika Serikat. Dataset ini sering digunakan dalam penelitian maupun pembelajaran machine learning untuk memprediksi harga rumah berdasarkan berbagai faktor lingkungan dan sosial ekonomi.

Dataset ini terdiri dari beberapa atribut yang menggambarkan kondisi lingkungan, tingkat ekonomi, serta karakteristik rumah di suatu wilayah.

#### Penjelasan Kolom pada Dataset

Dataset yang digunakan pada penelitian ini adalah **Boston Housing Dataset** yang diperoleh dari platform Kaggle. Dataset ini berisi berbagai atribut yang menggambarkan kondisi lingkungan, sosial ekonomi, serta karakteristik wilayah perumahan di kota Boston. Setiap kolom pada dataset memiliki arti sebagai berikut:

| Kolom | Penjelasan |
|------|-------------|
| **CRIM** | Tingkat kejahatan per kapita di setiap wilayah kota Boston. |
| **ZN** | Persentase lahan perumahan yang dialokasikan untuk lot besar (lebih dari 25.000 kaki persegi). |
| **INDUS** | Persentase lahan bisnis non-ritel di wilayah tersebut. |
| **CHAS** | Variabel dummy untuk sungai Charles (1 jika wilayah berbatasan dengan sungai, 0 jika tidak). |
| **NOX** | Konsentrasi nitrogen oksida yang menunjukkan tingkat polusi udara. |
| **RM** | Rata-rata jumlah ruangan dalam rumah. |
| **AGE** | Persentase rumah yang dibangun sebelum tahun 1940. |
| **DIS** | Jarak rata-rata ke lima pusat pekerjaan utama di Boston. |
| **RAD** | Indeks akses ke jalan raya utama. |
| **TAX** | Tarif pajak properti per \$10.000. |
| **PTRATIO** | Rasio jumlah murid terhadap guru di wilayah tersebut. |
| **B** | Indeks yang berkaitan dengan komposisi rasial wilayah dalam dataset. |
| **LSTAT** | Persentase penduduk dengan status sosial ekonomi rendah. |
| **MEDV** | Nilai median harga rumah dalam ribuan dolar. Biasanya digunakan sebagai variabel target dalam analisis. |

Setiap atribut pada dataset ini dapat digunakan untuk menganalisis hubungan antara kondisi lingkungan dan sosial ekonomi dengan harga rumah di kota Boston.
### 1.1. Perhitungan WKNN Manual
#### 1.1.1 Menentukan Data yang Memiliki Missing Value
Langkah pertama adalah mengidentifikasi baris data yang memiliki nilai yang hilang (missing value).  
Pada dataset Boston Housing, terdapat missing value pada atribut **LSTAT** yang akan dihitung menggunakan metode WKNN.

#### 1.1.2  Menentukan Kolom yang Akan Dinormalisasi
Sebelum menghitung similarity, dilakukan normalisasi data agar semua atribut memiliki skala yang sama.  
Kolom yang dinormalisasi adalah:

- CRIM  
- INDUS  
- NOX  
- RM  
- AGE  
- DIS  
- TAX  
- PTRATIO  
- B  
- LSTAT  

Sedangkan kolom **CHAS** dan **RAD** tidak dinormalisasi karena merupakan variabel kategori atau indeks.

#### 1.1.3 Melakukan Normalisasi Data
Normalisasi dilakukan menggunakan metode **Min-Max Normalization** dengan rumus:

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

|        | CRIM | INDUS | NOX | RM | AGE | DIS | TAX | PTRATIO | B | LSTAT | MEDV |
|--------|------|------|------|------|------|------|------|------|------|------|------|
| Min | 0.00632 | 2.18 | 0.458 | 6.421 | 45.8 | 4.09 | 222 | 15.3 | 392.83 | 2.94 | 21.6 |
| Max | 0.06905 | 7.07 | 0.538 | 7.185 | 78.9 | 6.0622 | 296 | 18.7 | 396.9 | 9.14 | 36.2 |

Table dengan nilai yang sudah dinormalisasi:
![original image](https://cdn.mathpix.com/snip/images/UvuF5Ik6z5AOPY_A853praI5dIYgojz-JdF5edkKu9I.original.fullsize.png)

#### 1.1.4 Menghitung Similarity (Si)
Setelah semua data dinormalisasi, langkah selanjutnya adalah menghitung nilai similarity antara data yang memiliki missing value dengan data lainnya.

Rumus similarity yang digunakan adalah:

$$
S_i = \frac{1}{\sum (Y_{ih} - Y_{jh})^2}
$$

Hasil perhitungan similarity yang diperoleh adalah sebagai berikut:

| Si |
|----|
| 0.096432567 |
| 0.186117946 |
| 0.251889237 |
| 1.262310597 |

#### 1.1.5 Menghitung Penyebut (ΣSi)
Nilai similarity kemudian dijumlahkan untuk mendapatkan nilai penyebut pada rumus imputasi.

$$
\sum S_i = 0.096432567 + 0.186117946 + 0.251889237 + 1.262310597
$$

$$
\sum S_i = 1.796750348
$$

#### 1.1.6 Menghitung Pembilang (Si × Yjh)
Selanjutnya dihitung perkalian antara nilai similarity dengan nilai atribut dari tetangga terdekat.

$$
S_i \times Y_{jh}
$$

Hasil dari setiap perkalian tersebut kemudian dijumlahkan untuk mendapatkan nilai pembilang.
| Si | Yjh | Si × Yjh |
|---|---|---|
| 0.096432567 | 0.329032258 | 0.031730 |
| 0.186117946 | 1 | 0.186118 |
| 0.251889237 | 0.175806452 | 0.044277 |
| 1.262310597 | 0 | 0 |

Jumlah pembilang:

$$
\sum (S_i \times Y_{jh}) = 0.262125
$$

#### 1.1.7 Hasil Nilai Imputasi

Nilai imputasi diperoleh dengan membagi pembilang dengan penyebut:

$$
\hat{y}_{ih} =
\frac{0.262125}{1.796750348}
$$

$$
\hat{y}_{ih} = 0.14589179
$$

Nilai tersebut merupakan hasil imputasi untuk menggantikan nilai **missing value pada atribut LSTAT**.


### 1.2 Perhitungan WKNN menggunakan Python

Selain melakukan perhitungan secara manual menggunakan Microsoft Excel, proses imputasi missing value juga dilakukan menggunakan bahasa pemrograman Python. Implementasi ini bertujuan untuk memverifikasi hasil perhitungan manual serta mempermudah proses pengolahan data.

Library yang digunakan dalam proses ini adalah:

- `pandas` untuk membaca dan mengolah dataset
- `numpy` untuk operasi numerik
- `sklearn` untuk membantu proses normalisasi data

#### 1. Mengunggah dan Membaca Dataset

Dataset terlebih dahulu diunggah ke Google Colab kemudian dibaca menggunakan library `pandas`.

```
from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances

df = pd.read_excel("data.xlsx")

df = df.replace("?", np.nan)
df = df.apply(pd.to_numeric)

print("DATA AWAL")
print(df)
```

**DATA AWAL**

| CRIM | INDUS | CHAS | NOX | RM | AGE | DIS | RAD | TAX | PTRATIO | B | LSTAT | MEDV |
|------|------|------|------|------|------|------|------|------|------|------|------|------|
| 0.00632 | 2.31 | 0 | 0.538 | 6.575 | 65.2 | 4.09 | 1 | 296 | 15.3 | 396.9 | 4.98 | 24.0 |
| 0.02731 | 7.07 | 0 | 0.469 | 6.421 | 78.9 | 4.97 | 2 | 242 | 17.8 | 396.9 | 9.14 | 21.6 |
| 0.02729 | 7.07 | 0 | 0.469 | 7.185 | 61.1 | 4.97 | 2 | 242 | 17.8 | 392.83 | 4.03 | 34.7 |
| 0.03237 | 2.18 | 0 | 0.458 | 6.998 | 45.8 | 6.06 | 3 | 222 | 18.7 | 394.63 | 2.94 | 33.4 |
| 0.06905 | 2.18 | 0 | 0.458 | 7.147 | 54.2 | 6.06 | 3 | 222 | 18.7 | 396.9 | ? | 36.2 |

#### 2.Menentukan Kolom yang Akan Dinormalisasi
Pada tahap ini ditentukan atribut yang akan digunakan dalam proses normalisasi.
```
cols_norm = [
'CRIM','INDUS','NOX','RM','AGE',
'DIS','TAX','PTRATIO','B','LSTAT','MEDV'
]
```
Selanjutnya dihitung nilai minimum dan maksimum dari setiap atribut.
```
min_val = df[cols_norm].min()
max_val = df[cols_norm].max()

print("MIN VALUE : ")
print(min_val)
print(30*'=')
print("MAX VALUE : ")
print(max_val)
```
OUTPUT:
```
MIN VALUE : 
CRIM         0.00632
INDUS        2.18000
NOX          0.45800
RM           6.42100
AGE         45.80000
DIS          4.09000
TAX        222.00000
PTRATIO     15.30000
B          392.83000
LSTAT        2.94000
MEDV        21.60000
dtype: float64
==============================
MAX VALUE : 
CRIM         0.06905
INDUS        7.07000
NOX          0.53800
RM           7.18500
AGE         78.90000
DIS          6.06220
TAX        296.00000
PTRATIO     18.70000
B          396.90000
LSTAT        9.14000
MEDV        36.20000
dtype: float64
==============================
```

#### 3. Melakukan Normalisasi Data
Normalisasi dilakukan menggunakan metode Min-Max Normalization
```
df_norm = df.copy()

for col in cols_norm:
    df_norm[col] = (df[col] - min_val[col]) / (max_val[col] - min_val[col])

print("DATA SETELAH NORMALISASI")
print(df_norm)
```
OUTPUT:
```
DATA SETELAH NORMALISASI
       CRIM     INDUS  CHAS     NOX        RM       AGE       DIS  RAD  \
0  0.000000  0.026585     0  1.0000  0.201571  0.586103  0.000000    1   
1  0.334609  1.000000     0  0.1375  0.000000  1.000000  0.444732    2   
2  0.334290  1.000000     0  0.1375  1.000000  0.462236  0.444732    2   
3  0.415272  0.000000     0  0.0000  0.755236  0.000000  1.000000    3   
4  1.000000  0.000000     0  0.0000  0.950262  0.253776  1.000000    3   

       TAX   PTRATIO        B     LSTAT      MEDV  
0  1.00000  0.000000  1.00000  0.329032  0.164384  
1  0.27027  0.735294  1.00000  1.000000  0.000000  
2  0.27027  0.735294  0.00000  0.175806  0.897260  
3  0.00000  1.000000  0.44226  0.000000  0.808219  
4  0.00000  1.000000  1.00000       NaN  1.000000 
```
#### 4. Menentukan Data yang Memiliki Missing Value
Selanjutnya dilakukan identifikasi data yang memiliki nilai kosong pada atribut LSTAT.
```
missing_index = df_norm[df_norm['LSTAT'].isna()].index
row_missing = df_norm.loc[missing_index].drop(columns=['LSTAT']).iloc[0]

complete_data = df_norm.dropna()
```
#### 5. Menghitung Nilai Similarity (Si)
```
Si = []
Yjh = []

for i in range(len(complete_data)):
    row = complete_data.iloc[i].drop('LSTAT')
    diff = (row_missing - row) ** 2
    sum_diff = diff.sum()
    similarity = 1 / sum_diff
    Si.append(similarity)
    Yjh.append(complete_data.iloc[i]['LSTAT'])

Si = np.array(Si)
Yjh = np.array(Yjh)

print(f"Nilai Si: {Si})
```
OUTPUT:
```
Nilai Si : [0.09643257 0.18611795 0.25188924 1.2623106 ]
```
#### 6. Menghitung Penyebut
Penyebut diperoleh dari jumlah seluruh nilai similarity.
```
penyebut = np.sum(Si)

print("Penyebut:")
print(penyebut)
```
OUTPUT:
```
Penyebut:
1.796750347642724
```
#### 7. Menghitung Pembilang
Pembilang diperoleh dari hasil perkalian similarity dengan nilai atribut tetangga.
```
Si_Yjh = Si * Yjh

print("Si * Yjh:", Si_Yjh)

pembilang = np.sum(Si_Yjh)

print("Pembilang:")
print(pembilang)
```
OUTPUT:
```
Si * Yjh:
[0.031730 0.186118 0.044277 0]

Pembilang:
0.262125
```
#### 8. Menghitung Nilai Imputasi
Nilai imputasi dihitung menggunakan rumus Weighted K-Nearest
```
hasil = pembilang / penyebut

print("Hasil Akhir:", hasil)
```
OUTPUT:
```
Hasil Akhir:
0.14589179
```
Nilai tersebut merupakan hasil imputasi yang digunakan untuk menggantikan nilai missing value pada atribut LSTAT sehingga dataset menjadi lengkap dan dapat digunakan untuk analisis selanjutnya.


## 2. Normalisasi data

Normalisasi data adalah proses mengubah skala nilai pada data sehingga berada pada rentang tertentu agar semua variabel memiliki skala yang sebanding. Normalisasi biasanya dilakukan pada tahap preprocessing data sebelum proses analisis atau pemodelan machine learning.

Tujuan normalisasi data adalah untuk menghindari dominasi nilai dari suatu variabel yang memiliki skala lebih besar dibandingkan variabel lainnya. Dengan normalisasi, setiap fitur dalam dataset dapat memberikan kontribusi yang seimbang dalam proses perhitungan.

### 2.1 Macam-macam Normalisasi Data

#### 1. MinMax Normalization
Min-Max Normalization adalah metode normalisasi yang mengubah nilai data ke dalam rentang tertentu, biasanya antara 0 hingga 1.

Rumus Min-Max Normalization:

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

$$
\begin{aligned}
x &= \text{nilai data asli} \\
x_{\min} &= \text{nilai minimum dalam dataset} \\
x_{\max} &= \text{nilai maksimum dalam dataset} \\
x' &= \text{nilai hasil normalisasi}
\end{aligned}
$$

**Contoh:**
Sebagai contoh digunakan sebagian data dari dataset Boston Housing pada kolom **RM**.

| RM |
|----|
| 6.575 |
| 6.421 |
| 7.185 |

Nilai minimum dan maksimum dari data tersebut adalah:

- \(x_{min} = 6.421\)
- \(x_{max} = 7.185\)

Perhitungan normalisasi untuk data pertama:

$$
x' = \frac{6.575 - 6.421}{7.185 - 6.421}
$$

$$
x' = \frac{0.154}{0.764}
$$

$$
x' = 0.201
$$

Hasil normalisasi:

| RM Asli | RM Normalisasi |
|-------|-------|
| 6.575 | 0.201 |
| 6.421 | 0 |
| 7.185 | 1 |

#### 2. Z-Score Normalization
Z-Score Normalization atau standardization adalah metode normalisasi yang menggunakan nilai rata-rata (mean) dan standar deviasi dari data.
Rumus Z-Score Normalization:

$$
x' = \frac{x - \mu}{\sigma}
$$


$$
\begin{aligned}
x &= \text{nilai data asli} \\
\mu &= \text{rata-rata (mean) dari data} \\
\sigma &= \text{standar deviasi} \\
x' &= \text{nilai hasil normalisasi}
\end{aligned}
$$

**Contoh:**
Menggunakan data yang sama pada kolom **RM**:


| RM |
|----|
| 6.575 |
| 6.421 |
| 7.185 |
##### Menghitung Rata-rata
Rumus mean:

$$
\mu = \frac{\sum x}{n}
$$

Perhitungan:

$$
\mu = \frac{6.575 + 6.421 + 7.185}{3}
$$

$$
\mu = \frac{20.181}{3}
$$

$$
\mu = 6.727
$$

##### Menghitung Standar Deviasi
Rumus standar deviasi:

$$
\sigma = \sqrt{\frac{\sum (x - \mu)^2}{n}}
$$

Perhitungan:

$$
(6.575 - 6.727)^2 = (-0.152)^2 = 0.0231
$$

$$
(6.421 - 6.727)^2 = (-0.306)^2 = 0.0936
$$

$$
(7.185 - 6.727)^2 = (0.458)^2 = 0.2097
$$

Jumlahkan:

$$
0.0231 + 0.0936 + 0.2097 = 0.3264
$$

Bagi dengan jumlah data:

$$
\frac{0.3264}{3} = 0.1088
$$

Akar kuadrat:

$$
\sigma = \sqrt{0.1088}
$$

$$
\sigma = 0.33
$$

Diketahui:

- Mean (\(\mu\)) = 6.727  
- Standar deviasi (\(\sigma\)) = 0.33  

Perhitungan untuk data pertama:

$$
x' = \frac{6.575 - 6.727}{0.33}
$$

$$
x' = -0.4606
$$

Hasil normalisasi:

| RM Asli | Z-Score |
|-------|-------|
| 6.575 | -0.4606 |
| 6.421 | -0.9272 |
| 7.185 | 1.3878 |

#### 3. Decimal Scaling Normalization
Decimal Scaling Normalization adalah metode normalisasi dengan cara memindahkan titik desimal dari nilai data.
Rumus Decimal Scaling:

$$
x' = \frac{x}{10^j}
$$

$$
\begin{aligned}
x &= \text{nilai data asli} \\
j &= \text{jumlah digit maksimum dalam data} \\
x' &= \text{nilai hasil normalisasi}
\end{aligned}
$$

**Contoh:**

Menggunakan data yang sama pada kolom **RM**.

| RM |
|----|
| 6.575 |
| 6.421 |
| 7.185 |

Nilai terbesar dari data tersebut adalah:

$$
7.185
$$

Karena nilai terbesar memiliki **1 digit sebelum desimal**, maka:

$$
j = 1
$$

Perhitungan untuk data pertama:

$$
x' = \frac{6.575}{10^1}
$$

$$
x' = 0.6575
$$

Hasil normalisasi:

| RM Asli | Decimal Scaling |
|-------|-------|
| 6.575 | 0.6575 |
| 6.421 | 0.6421 |
| 7.185 | 0.7185 |

### 2.2 Normalisasi Data Menggunakan SK_Learn
Selain melakukan perhitungan normalisasi secara manual, normalisasi data juga dapat dilakukan menggunakan library Python yaitu **scikit-learn (sklearn)**. Library ini menyediakan berbagai fungsi preprocessing data yang memudahkan proses normalisasi sebelum data digunakan dalam proses analisis atau machine learning.

Pada contoh berikut digunakan sebagian data dari **Boston Housing Dataset** pada kolom **RM**.

### Data Awal

| RM |
|------|
| 6.575 |
| 6.421 |
| 7.185 |

#### 1. Min-Max Normalization menggunakan sklearn

Min-Max Normalization dapat dilakukan menggunakan **MinMaxScaler** dari sklearn.

**Kode Python**
```
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {'RM':[6.575,6.421,7.185]}
df = pd.DataFrame(data)

scaler = MinMaxScaler()

df['RM_MinMax'] = scaler.fit_transform(df[['RM']])

print(df)
```

OUPUT :
| RM    | RM_MinMax |
| ----- | --------- |
| 6.575 | 0.201     |
| 6.421 | 0         |
| 7.185 | 1         |


#### 2. Z-Score Normalization menggunakan sklearn
Z-Score Normalization dapat dilakukan menggunakan **StandardScaler** dari sklearn.

**Kode Python**
```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df['RM_ZScore'] = scaler.fit_transform(df[['RM']])

print(df)
```
OUTPUT:
| RM    | RM_ZScore |
| ----- | --------- |
| 6.575 | -0.4606   |
| 6.421 | -0.9272   |
| 7.185 | 1.3878    |

#### 3. Decimal Scaling menggunakan Python
Library sklearn tidak menyediakan fungsi khusus untuk Decimal Scaling. Oleh karena itu normalisasi ini dapat dilakukan dengan perhitungan sederhana menggunakan Python.
**Kode Python**
```
import numpy as np

data = {'RM':[6.575,6.421,7.185]}
df = pd.DataFrame(data)

max_value = df['RM'].max()

j = len(str(int(max_value)))

df['RM_Decimal'] = df['RM'] / (10**j)

print(df)
```
OUTPUT:
| RM    | RM_Decimal |
| ----- | ---------- |
| 6.575 | 0.6575     |
| 6.421 | 0.6421     |
| 7.185 | 0.7185     |


Berdasarkan hasil implementasi menggunakan sklearn, nilai normalisasi yang dihasilkan sama dengan perhitungan manual sebelumnya. Hal ini menunjukkan bahwa library sklearn dapat digunakan untuk mempermudah proses normalisasi data secara otomatis.