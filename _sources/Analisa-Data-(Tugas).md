# Analisa Data (dokumentasi Skforecast)

#### 1. Analisis prediksi tentang apa?

Analisis prediksi pada dokumentasi Skforecast tersebut bertujuan untuk meramalkan permintaan atau konsumsi listrik (electricity demand) di wilayah Victoria, Australia pada periode waktu berikutnya. Prediksi dilakukan menggunakan pendekatan time series forecasting, yaitu memanfaatkan pola data historis untuk memperkirakan nilai di masa depan.

Data yang digunakan tidak hanya berupa nilai konsumsi listrik pada periode-periode sebelumnya, tetapi juga mempertimbangkan faktor eksternal seperti suhu (temperature) yang dapat memengaruhi kebutuhan listrik. Misalnya, ketika suhu terlalu panas atau terlalu dingin, penggunaan pendingin ruangan atau pemanas cenderung meningkat sehingga permintaan listrik juga bertambah.

Dalam prosesnya, model mempelajari hubungan antara:
* Konsumsi listrik pada beberapa periode sebelumnya (lag),
* Suhu lingkungan,
* Pola perubahan konsumsi listrik dari waktu ke waktu,
untuk menghasilkan prediksi konsumsi listrik pada periode berikutnya.

Selain menghasilkan prediksi, analisis ini juga berfokus pada explainability (interpretasi model). Oleh karena itu, digunakan metode seperti Feature Importance dan SHAP (SHapley Additive exPlanations) untuk mengetahui faktor-faktor yang paling memengaruhi hasil prediksi. Dengan cara ini, pengguna tidak hanya mengetahui hasil prediksi, tetapi juga dapat memahami alasan model menghasilkan prediksi tersebut.

#### 2. Bagaimana bentuk data trainingnya (apa saja input dan outputnya)?

Data training pada analisis ini berbentuk supervised learning, yaitu terdiri dari data input (X) dan data output atau target (y). Model dilatih menggunakan data historis permintaan listrik (Demand) dan data suhu (Temperature) untuk memprediksi nilai permintaan listrik pada periode berikutnya.

Hal ini dapat dilihat pada kode berikut:
```
forecaster.fit(
    y    = data_train['Demand'],
    exog = data_train['Temperature']
)
```
Pada kode tersebut:
* y = data_train['Demand'] menunjukkan bahwa target atau output yang diprediksi adalah Demand (permintaan listrik).
* exog = data_train['Temperature'] menunjukkan bahwa Temperature (suhu) digunakan sebagai variabel input tambahan.

Selain itu, bentuk data training dapat dilihat dengan kode:
```
X_train, y_train = forecaster.create_train_X_y(
    y    = data_train['Demand'],
    exog = data_train['Temperature']
)

display(X_train.head(3))
display(y_train.head(3))
```
Model juga menggunakan:
```
forecaster = ForecasterRecursive(
    regressor = LGBMRegressor(random_state=123, verbose=-1),
    lags = 7
)
```
Parameter lags = 7 berarti model menggunakan 7 data Demand sebelumnya (lag 1 sampai lag 7) sebagai fitur input.

Table Output:




#### 3. Apa itu lag?

Lag adalah nilai data pada periode sebelumnya yang digunakan sebagai input untuk memprediksi nilai pada periode berikutnya. Dalam analisis time series, lag berfungsi untuk menangkap pola historis yang dapat memengaruhi nilai di masa depan.

Pada kode yang digunakan:
```
forecaster = ForecasterRecursive(
    regressor = LGBMRegressor(random_state=123, verbose=-1),
    lags = 7
)
```
nilai lags = 7 menunjukkan bahwa model menggunakan 7 data permintaan listrik (Demand) sebelumnya sebagai dasar untuk memprediksi permintaan listrik berikutnya.


#### 4. Jelaskan proses analisis yang dilakukan dari kasus di atas

Proses analisis yang dilakukan pada kasus ini bertujuan untuk memprediksi permintaan listrik (Demand) di Victoria, Australia menggunakan data historis dan suhu (Temperature). Tahapan analisis yang dilakukan adalah sebagai berikut:
1. Mengunduh dataset
Dataset Vic Electricity diambil menggunakan fungsi:
```
data = fetch_dataset(name="vic_electricity")
```
Dataset berisi informasi permintaan listrik (Demand) dan suhu (Temperature).

2. Melakukan agregasi data

Data diubah menjadi data harian menggunakan:
```
data = data.resample('D').agg({
    'Demand': 'sum',
    'Temperature': 'mean'
})
```
Demand dijumlahkan per hari, sedangkan Temperature dihitung rata-ratanya.

3. Demand dijumlahkan per hari, sedangkan Temperature dihitung rata-ratanya.
```
data_train = data.loc[:'2014-12-21']
data_test = data.loc['2014-12-22':]
```
Data training digunakan untuk melatih model, sedangkan data testing digunakan untuk pengujian.

4. Membangun model forecasting

Model dibuat menggunakan algoritma LightGBM dan metode ForecasterRecursive dengan 7 lag:
```
forecaster = ForecasterRecursive(
    regressor=LGBMRegressor(random_state=123, verbose=-1),
    lags=7
)
```

5. Melatih model

Model dilatih menggunakan data Demand sebagai target dan Temperature sebagai variabel tambahan:
```
forecaster.fit(
    y=data_train['Demand'],
    exog=data_train['Temperature']
)
```

6. Menganalisis feature importance
```
forecaster.get_feature_importances()
```

7. Melakukan explainability menggunakan SHAP
```
explainer = shap.TreeExplainer(forecaster.regressor)
shap_values = explainer.shap_values(X_train)
```
SHAP digunakan untuk menjelaskan kontribusi masing-masing fitur terhadap prediksi yang dihasilkan model.

8. Visualisasi hasil explainability
```
shap.summary_plot(shap_values, X_train)

shap.force_plot(
    explainer.expected_value,
    shap_values[0, :],
    X_train.iloc[0, :]
)
```

Visualisasi ini membantu memahami fitur yang paling berpengaruh dalam proses prediksi.