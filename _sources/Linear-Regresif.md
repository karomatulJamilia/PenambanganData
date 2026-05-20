# Linear Regresision

## 1. Linear Regresi
Regresi linier  adalah metode statistik yang digunakan untuk mengetahui hubungan antara variabel bebas $x$ dan variabel terikat $y$.

Pada tugas ini digunakan metode regresi linier untuk mencari:

- $b_0$ = intercept
- $b_1$ = koefisien regresi

Persamaan umum regresi linier:

$$
y = b_0 + b_1x
$$

Keterangan:

- $y$ = variabel terikat
- $x$ = variabel bebas
- $b_0$ = intercept
- $b_1$ = slope / koefisien regresi


## 2. Dataset

Data titik yang digunakan:

| Titik | x | y |
|---|---|---|
| A | 2 | 2 |
| B | 4 | 3 |
| C | 3 | 5 |
| D | 3 | 4 |
| E | 3 | 3 |
| F | 4 | 5 |
| G | 5 | 6 |


## 3. Library yang Digunakan

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
```

Penjelasan:

- `numpy` digunakan untuk perhitungan numerik dan matriks.
- `LinearRegression` digunakan untuk membuat model regresi linier.
- `matplotlib` digunakan untuk visualisasi grafik.


## 4. Source Code Program

```
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import metrics
import statsmodels.formula.api as smf

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# =========================
# DATA
# =========================

x = np.array([2,4,3,3,3,4,5]).reshape(-1,1)
y = np.array([2,3,5,4,3,5,6])

# =========================
# MODEL REGRESI
# =========================

model = LinearRegression()

# training model
model.fit(x, y)

# =========================
# KOEFISIEN REGRESI
# =========================

# b1 = slope / koefisien
b1 = model.coef_[0]

# b0 = intercept
b0 = model.intercept_

# =========================
# OUTPUT
# =========================

print("Nilai b1 =", b1)
print("Nilai b0 =", b0)

print(f"Persamaan regresi: y = {b1:.2f}x + {b0:.2f}")

# =========================
# PREDIKSI
# =========================

x_baru = 2

y_prediksi = model.predict([[x_baru]])

print(f"Prediksi y untuk x={x_baru} adalah {y_prediksi[0]}")

# =========================
# VISUALISASI
# =========================

plt.scatter(x, y, color='blue', label='Data Asli')

plt.plot(x,
         model.predict(x),
         color='black',
         label='Garis Regresi')

plt.xlabel("X")
plt.ylabel("Y")

plt.title("Linear Regression")

plt.legend()

plt.grid(True)

plt.show()
```



## 5. Hasil Program
![original image](https://cdn.mathpix.com/snip/images/odLMtQaUqCgZkIyskWzhWwfsI6aTd67QDq7W0UDaBg0.original.fullsize.png)

Output program:

```python
Nilai b1 = 1.05
Nilai b0 = 0.4

Persamaan regresi:
y = 1.05x + 0.4
```

Penjelasan:

- Nilai $b_1 = 1.05$ menunjukkan bahwa setiap kenaikan 1 nilai $x$ akan meningkatkan nilai $y$ sebesar 1.05.
- Nilai $b_0 = 0.4$ merupakan titik potong garis regresi terhadap sumbu-Y.
- Jika x = 2,  maka model regresi memprediksi nilai Y sebesar 2.5.

## 6. Perhitungan Manual Menggunakan Matriks

Koefisien regresi linier dapat dihitung menggunakan persamaan matriks:

$$
B = (X^TX)^{-1}X^TY
$$

Keterangan:

- $B$ = vektor koefisien regresi
- $X^T$ = transpose matriks $X$
- $(X^TX)^{-1}$ = invers matriks
- $Y$ = matriks target/output

Bentuk vektor koefisien regresi:

$$
B =
\begin{bmatrix}
b_0 \\
b_1
\end{bmatrix}
$$

Dengan:

- $b_0$ = intercept
- $b_1$ = koefisien regresi

### Matriks X

$$
X =
\begin{bmatrix}
1 & 2 \\
1 & 4 \\
1 & 3 \\
1 & 3 \\
1 & 3 \\
1 & 4 \\
1 & 5
\end{bmatrix}
$$

### Matriks Y

$$
Y =
\begin{bmatrix}
2 \\
3 \\
5 \\
4 \\
3 \\
5 \\
6
\end{bmatrix}
$$

### Menghitung $X^TX$

$$
X^TX =
\begin{bmatrix}
7 & 24 \\
24 & 88
\end{bmatrix}
$$

Penjelasan:

$$
X^TX =
\begin{bmatrix}
1 & 1 & 1 & 1 & 1 & 1 & 1 \\
2 & 4 & 3 & 3 & 3 & 4 & 5
\end{bmatrix}
\begin{bmatrix}
1 & 2 \\
1 & 4 \\
1 & 3 \\
1 & 3 \\
1 & 3 \\
1 & 4 \\
1 & 5
\end{bmatrix}
$$

Hasil perkalian:



$\begin{bmatrix}
7 & 24 \\
24 & 88
\end{bmatrix}$


### Menghitung Invers Matriks

Determinan matriks:


$(7 \times 88) - (24 \times 24)$



$= 616 - 576$

$= 40$

Maka invers matriks:

$$ (X^TX)^{-1} = \frac{1}{40} \left[ \begin{array}{cc} 88 & -24 \\ -24 & 7 \end{array} \right] $$

### Menghitung $X^TY$

$$
X^TY =
\begin{bmatrix}
1 & 1 & 1 & 1 & 1 & 1 & 1 \\
2 & 4 & 3 & 3 & 3 & 4 & 5
\end{bmatrix}
\begin{bmatrix}
2 \\
3 \\
5 \\
4 \\
3 \\
5 \\
6
\end{bmatrix}
$$

Hasil:

$$
X^TY =
\begin{bmatrix}
28 \\
102
\end{bmatrix}
$$

### Menghitung Koefisien Regresi

Substitusi ke persamaan regresi:

$$ B = (X^TX)^{-1}X^TY $$

$$ B = \frac{1}{40} \left[ \begin{array}{cc} 88 & -24 \\ -24 & 7 \end{array} \right] \left[ \begin{array}{c} 28 \\ 102 \end{array} \right] $$

Hasil perkalian:

$$ = \frac{1}{40} \left[ \begin{array}{c} 16 \\ 42 \end{array} \right] $$
Hasil akhir:

$$
B =
\begin{bmatrix}
0.4 \\
1.05
\end{bmatrix}
$$

Sehingga diperoleh:

$$
b_0 = 0.4
$$

$$
b_1 = 1.05
$$

### Persamaan Regresi

Persamaan regresi linier yang diperoleh:

$$
y = 1.05x + 0.4
$$

Keterangan:

- $1.05$ adalah koefisien regresi ($b_1$)
- $0.4$ adalah intercept ($b_0$)

### Contoh Prediksi

Jika:

$$
x = 2
$$

Maka:

$$
y = 1.05(2) + 0.4
$$

$$
= 2.1 + 0.4
$$

$$
= 2.5
$$

Artinya ketika nilai $x = 2$, model memprediksi nilai $y = 2.5$.

##  7.Visualisasi Menggunakan GeoGebra

Berikut merupakan visualisasi regresi linier menggunakan GeoGebra berdasarkan data yang digunakan.

![original image](https://cdn.mathpix.com/snip/images/c_irT0s0yk4RBdUmn8R3uF9B5FNrVoxlwS_E2WC3vvQ.original.fullsize.png)


Penjelasan:

- Titik biru menunjukkan data asli.
- Garis hitam menunjukkan garis regresi linier.
- Persamaan garis regresi yang diperoleh adalah:

$$
y = 1.05x + 0.4
$$

- Garis regresi memiliki arah positif, sehingga menunjukkan bahwa variabel $x$ dan $y$ memiliki hubungan positif.
- Semakin besar nilai $x$, maka nilai $y$ cenderung meningkat.
- Titik $H$ menunjukkan intercept sebesar $0.4$ pada sumbu-$Y$.
- Titik $I$ menunjukkan hasil prediksi saat $x = 2$, yaitu:

$$
y = 2.5
$$