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

