
# Crawling Data NO2

## Prediksi Konsentrasi NO₂ di Wilayah Kecamatan Batang-Batang, Kabupaten Sumenep, Madura


# 1. Latar Belakang

Pencemaran udara merupakan salah satu permasalahan lingkungan yang semakin mendapat perhatian karena berdampak langsung terhadap kesehatan manusia, ekosistem, dan kualitas hidup masyarakat. Salah satu polutan udara yang sering digunakan sebagai indikator kualitas udara adalah Nitrogen Dioksida (NO₂). Gas NO₂ dihasilkan terutama dari proses pembakaran bahan bakar fosil, seperti aktivitas transportasi, industri, pembangkit listrik, serta pembakaran biomassa. Paparan NO₂ dalam konsentrasi tinggi dapat menyebabkan gangguan sistem pernapasan, menurunkan fungsi paru-paru, dan meningkatkan risiko berbagai penyakit pernapasan.

Kabupaten Sumenep sebagai salah satu wilayah di Pulau Madura mengalami perkembangan aktivitas ekonomi, transportasi, dan pemukiman yang berpotensi memengaruhi kualitas udara. Meskipun tingkat industrialisasi di wilayah ini tidak setinggi daerah perkotaan besar, peningkatan aktivitas manusia dapat menyebabkan perubahan konsentrasi polutan udara, termasuk NO₂. Oleh karena itu, pemantauan dan analisis konsentrasi NO₂ menjadi penting untuk mengetahui kondisi kualitas udara serta sebagai dasar dalam pengambilan keputusan terkait pengelolaan lingkungan.

Perkembangan teknologi penginderaan jauh memungkinkan pemantauan kualitas udara dilakukan secara lebih luas dan efisien. Salah satu sumber data yang banyak digunakan adalah satelit Sentinel-5P yang menyediakan informasi konsentrasi berbagai gas atmosfer, termasuk Nitrogen Dioksida (NO₂). Data satelit memiliki keunggulan dalam cakupan wilayah yang luas, ketersediaan data secara berkala, serta kemudahan akses sehingga dapat dimanfaatkan untuk menganalisis kondisi kualitas udara pada suatu wilayah.

Selain mengetahui kondisi konsentrasi NO₂ pada masa lalu dan saat ini, diperlukan juga upaya untuk memperkirakan kondisi pada masa mendatang melalui proses peramalan (forecasting). Peramalan konsentrasi NO₂ dapat memberikan gambaran mengenai tren perubahan kualitas udara sehingga dapat digunakan sebagai langkah antisipatif dalam pengelolaan lingkungan dan kesehatan masyarakat. Dengan adanya hasil peramalan, pihak terkait dapat menyusun strategi yang lebih efektif dalam mengendalikan sumber pencemaran udara dan meminimalkan dampak yang ditimbulkan.

Berdasarkan uraian tersebut, penelitian ini dilakukan untuk menganalisis dan meramalkan konsentrasi Nitrogen Dioksida (NO₂) di wilayah Kabupaten Sumenep, Madura menggunakan data penginderaan jauh dari satelit Sentinel-5P. Hasil penelitian diharapkan dapat memberikan informasi mengenai pola perubahan konsentrasi NO₂ serta menjadi referensi dalam upaya pemantauan dan pengelolaan kualitas udara di wilayah Kabupaten Sumenep.

# 2. Pengumpulan Data

Data yang digunakan dalam penelitian ini adalah data konsentrasi Nitrogen Dioksida (NO₂) harian yang diperoleh dari satelit Sentinel-5P milik program Copernicus. Sentinel-5P merupakan satelit pengamatan atmosfer yang menyediakan berbagai informasi kualitas udara, termasuk konsentrasi gas Nitrogen Dioksida (NO₂) pada skala global.

Pengambilan data dilakukan melalui platform Copernicus Data Space Ecosystem yang dapat diakses melalui website:
https://dataspace.copernicus.eu/

Sebelum proses pengambilan data dilakukan, pengguna harus membuat akun pada platform tersebut untuk memperoleh akses ke layanan pengolahan data. Selanjutnya, proses ekstraksi data dilakukan menggunakan bahasa pemrograman Python pada lingkungan Google Colab dengan memanfaatkan API OpenEO yang disediakan oleh Copernicus. 

Pengambilan data mengacu pada dokumentasi resmi Copernicus Data Space Ecosystem:

Dokumentasi Pengambilan Data NO₂ Sentinel-5P
https://documentation.dataspace.copernicus.eu/notebook-samples/openeo/NO2Covid.html?utm_source=chatgpt.com

Hasil ekstraksi kemudian disimpan dalam format tabel (CSV) yang berisi informasi tanggal pengamatan dan nilai rata-rata konsentrasi NO₂ pada wilayah penelitian. Data tersebut selanjutnya digunakan sebagai data deret waktu (time series) untuk proses analisis dan peramalan.
**Instalasi Library OpenEO**
Tahap pertama adalah menginstal library OpenEO pada Google Colab menggunakan perintah:
```
pip install openeo
```

Library OpenEO digunakan untuk menghubungkan Google Colab dengan platform Copernicus Data Space Ecosystem sehingga data satelit dapat diakses dan diolah secara langsung melalui Python.

**Koneksi ke Copernicus Data Space Ecosystem**
Setelah library berhasil diinstal, dilakukan koneksi ke server Copernicus Data Space Ecosystem menggunakan kode berikut:
```
import openeo

connection = openeo.connect(
    "openeo.dataspace.copernicus.eu").authenticate_oidc()
```

Proses autentikasi dilakukan menggunakan akun Copernicus yang telah dibuat sebelumnya. Setelah autentikasi berhasil, pengguna dapat mengakses berbagai dataset yang tersedia pada platform tersebut.

**Penentuan Area of Interest (AOI)**
Area penelitian ditentukan menggunakan koordinat polygon yang mewakili wilayah penelitian di Kabupaten Sumenep, Madura.
```
aoi = {
    "type": "Polygon",
    "coordinates": [
        [
            [113.98149077321028, -6.948874833195404],
            [114.04317937610563, -6.948835353358092],
            [114.0431404450419, -6.977611503812142],
            [113.9814777326718, -6.977613801674067],
            [113.98149077321028, -6.948874833195404]
          ]
    ]
}

```
AOI digunakan untuk membatasi wilayah pengambilan data sehingga hanya data yang berada pada area penelitian yang akan diproses.
| Titik | Longitude          | Latitude           |
| ----- | ------------------ | ------------------ |
| 1     | 113.98149077321028 | -6.948874833195404 |
| 2     | 114.04317937610563 | -6.948835353358092 |
| 3     | 114.04314044504190 | -6.977611503812142 |
| 4     | 113.98147773267180 | -6.977613801674067 |

**Pengambilan Data Sentinel-5P NO₂**
Dataset yang digunakan adalah Sentinel-5P Level-2 dengan parameter Nitrogen Dioksida (NO₂).
```
s5post = connection.load_collection(
    "SENTINEL_5P_L2",
    temporal_extent=["2023-10-01", "2025-10-01"],
    spatial_extent={
        "west": 112.68,
        "south": -7.20,
        "east": 113.09,
        "north": -6.89
    },
    bands=["NO2"],
)
```

Data diambil pada periode 1 Oktober 2023 sampai 1 Oktober 2025. Parameter yang dipilih adalah band NO₂ yang merepresentasikan konsentrasi Nitrogen Dioksida di atmosfer.

**Agregasi Temporal Harian**
Karena data satelit dapat memiliki lebih dari satu observasi dalam sehari, dilakukan agregasi temporal menggunakan rata-rata harian.
```
s5p_no2_daily = s5post.aggregate_temporal_period(
    reducer="mean",
    period="day"
)
```
Proses ini menghasilkan satu nilai rata-rata NO₂ untuk setiap hari pengamatan.

**Agregasi Spasial pada AOI**
Setelah agregasi temporal, dilakukan agregasi spasial untuk memperoleh nilai rata-rata NO₂ pada seluruh wilayah penelitian.
```
s5p_no2_aoi = s5p_no2_daily.aggregate_spatial(
    reducer="mean",
    geometries=aoi
)
```
Metode mean digunakan untuk menghitung rata-rata konsentrasi NO₂ pada seluruh area penelitian.

Code diatas memerlukan titik koordinasi area yang akan diambil data $NO2$
-nya, untuk mengambil titik koordinasi kaian kunjungi webiste https://geojson.io/#map=14.8/-7.04732/112.69463 . Didalam website tersebut kalian akan memilih daerah dengan cara memberi shape kotak didaerah yang ingin kalian ambil datanya.
![original image](https://cdn.mathpix.com/snip/images/K8BjMTh1QtdOPSOuj2waqfWYCf_QokX4Z_tWDobpGXM.original.fullsize.png)
Di panel sebelah kanan terdapat data JSON yang berupa koordinat daerah yang kalian pilih, kalian salin terus sesuaikan dengan code diatas di bagian variabel “aoi” dan spatial_extent.


**Eksekusi dan Penyimpanan Data**
Data kemudian diproses dan disimpan dalam format NetCDF (.nc).
```
job = s5post.execute_batch(
    title="NO2 in Sumenep",
    outputfile="NO2Sumenep.nc"
)
```
Format NetCDF dipilih karena mampu menyimpan data multidimensi yang umum digunakan pada data satelit dan penginderaan jauh.
Tunggu proses pengambilan data, output proses seperti berikut:
```
0:00:00 Job 'j-26060303355647c79a825545b3e31be9': send 'start'
0:00:10 Job 'j-26060303355647c79a825545b3e31be9': queued (progress 0%)
0:00:15 Job 'j-26060303355647c79a825545b3e31be9': queued (progress 0%)
0:00:21 Job 'j-26060303355647c79a825545b3e31be9': queued (progress 0%)
0:00:30 Job 'j-26060303355647c79a825545b3e31be9': queued (progress 0%)
0:00:40 Job 'j-26060303355647c79a825545b3e31be9': queued (progress 0%)
0:00:52 Job 'j-26060303355647c79a825545b3e31be9': queued (progress 0%)
0:01:08 Job 'j-26060303355647c79a825545b3e31be9': queued (progress 0%)
0:01:27 Job 'j-26060303355647c79a825545b3e31be9': running (progress N/A)
0:01:51 Job 'j-26060303355647c79a825545b3e31be9': running (progress N/A)
0:02:22 Job 'j-26060303355647c79a825545b3e31be9': running (progress N/A)
0:02:59 Job 'j-26060303355647c79a825545b3e31be9': running (progress N/A)
0:03:46 Job 'j-26060303355647c79a825545b3e31be9': running (progress N/A)
0:04:45 Job 'j-26060303355647c79a825545b3e31be9': running (progress N/A)
0:05:45 Job 'j-26060303355647c79a825545b3e31be9': running (progress N/A)
0:06:45 Job 'j-26060303355647c79a825545b3e31be9': finished (progress 100%)
```
Abaikan ketika ada N/A.

Ketika proses pengambilan data, aktivitas kalian akan terekam di halaman https://editor.openeo.org/?server=https%3A%2F%2Fopeneo.dataspace.copernicus.eu%2Fopeneo%2F1.2 . Disitu terdapat nama dataset dan status pengambilan data.
![original image](https://cdn.mathpix.com/snip/images/3sPXYypEHIwrVF3SmqY9a3u5F9x0U3WiPFAsO4U52Fw.original.fullsize.png)

# 3. Preproccessing Data
Setelah kita mengambil data, data bisa diunduh di halaman https://editor.openeo.org/?server=https%3A%2F%2Fopeneo.dataspace.copernicus.eu%2Fopeneo%2F1.2 . File akan berbentuk .nc.

**Ekstraksi Data NetCDF**
Untuk membaca file hasil unduhan, digunakan library NetCDF4.
```
pip install netCDF4
```
Kemudian file dibuka menggunakan:
```
import netCDF4

file_path = "NO2Sumenep.nc"
ds = netCDF4.Dataset(file_path)

# Lihat seluruh variabel yang tersedia
print("Variabel dalam file:")
print(ds.variables.keys())
# dict_keys(['t', 'x', 'y', 'crs', 'NO2'])

# Ambil NO2
no2 = ds.variables["NO2"][:]

# Ambil Time
time = ds.variables["t"][:]

# Konversi waktu ke format tanggal jika punya atribut 'units'
try:
    time_units = ds.variables["t"].units
    dates = netCDF4.num2date(time, units=time_units)
except Exception:
    dates = time  # fallback kalau tidak ada units

# Tampilkan struktur data NO2
print(type(no2))
# type <class 'numpy.ma.core.MaskedArray'>

print(len(no2))
# banyaknya data record NO2 725

print(len(no2[0]))
# panjang data perbaris 9

print(len(no2[0][0]))
# panjang perdata 8

print(no2[0][0][0])
# 3.7701793e-05
```
Dari code diatas kita mengetahui bentuk data dari kolom NO2 nya.

jadi struktur data NO2 perbaris adalah:

```
[
    [[] * 8] * 9
]
```

Untuk melihat 10 data pertama adalah:

```
print("Contoh data pertama:")
for i in range(0, 10):
    print(no2[i])
```

**Mengatasi Missing Value menggunakan metode Interpolasi Linear**
Setelah data time series terbentuk, dilakukan pemeriksaan untuk mengetahui adanya tanggal yang tidak memiliki data pengamatan. Tanggal yang hilang kemudian ditambahkan kembali ke dalam dataset dan nilai NO₂ yang kosong diisi menggunakan metode interpolasi waktu (time interpolation). Selanjutnya, nilai kosong yang masih tersisa pada awal atau akhir data diisi menggunakan metode forward fill dan backward fill sehingga diperoleh data harian yang lengkap tanpa missing value.
```
import numpy as np
import pandas as pd

# Interpolasi Linear
no2_filled = np.zeros_like(no2)
# Untuk jaga-jaga jika terdapat '--' tidak berubah menjadi 0
no2_filled = no2_filled.filled(0)

# loop tiap grid (y,x)
for i in range(no2.shape[1]):     # 9 baris
    for j in range(no2.shape[2]): # 8 kolom
        series = pd.Series(no2[:, i, j])
        no2_filled[:, i, j] = series.interpolate(method='linear', limit_direction='both').to_numpy()


new_dates = []
new_no2 = []
for i in range(len(dates)):
    # ubah format datetime
    new_date = dates[i].strftime('%Y-%m-%d')
    new_dates.append(new_date)
    new_no2.append(np.mean(no2_filled[i]))
```
**Simpan data ke format CSV**
```
df = pd.DataFrame({
    "date": new_dates,
    "NO2": new_no2
})

# Simpan ke CSV
df.to_csv("NO2_Sumenep_timeseries.csv", index=False)
```

**Pengecekan Missing Value data harian pada CSV**
```
import pandas as pd
import numpy as np

df = pd.read_csv("NO2_Sumenep_timeseries.csv")

# Pastikan kolom 'date' bertipe datetime
df['date'] = pd.to_datetime(df['date'])

# Buat rentang tanggal lengkap
start_date = "2023-10-01"
end_date = "2025-09-30"
full_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Cek tanggal yang hilang
missing_dates = full_range.difference(df['date'])

print(f"Jumlah hari missing: {len(missing_dates)}")
print("Daftar tanggal missing:")
print(missing_dates)
```
output:
```
Jumlah hari missing: 0
Daftar tanggal missing:
DatetimeIndex([], dtype='datetime64[ns]', freq='D')
```
Dalam kasus saya ini, terdapat 6 hari missing value. Kita akan mengatasi lagi missing value menggunakan metode Interpolasi Linear. Cara memperbaikinya gunakan code dibawah:
```
import pandas as pd

# Pastikan datetime dan sorting
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Buat rentang tanggal lengkap
full_range = pd.date_range(start="2023-10-01", end="2025-09-30", freq='D')

# Reindex agar tanggal yang hilang muncul sebagai NaN
df = df.set_index('date').reindex(full_range)
df.index.name = 'date'

# Interpolasi linear berdasarkan indeks waktu
df['NO2'] = df['NO2'].interpolate(method='time')

# (Opsional) jika masih ada NaN di bagian awal/akhir bisa gunakan forward/backward fill
df['NO2'] = df['NO2'].fillna(method='bfill').fillna(method='ffill')

# Simpan kembali ke CSV
df.to_csv("no2_timeseries_interpolated.csv")
```

**Deteksi Outlier IQR**
Setelah kita mengisi missing value menggunakan metode Interpolasi Linear, selanjutnya kita akan mendeteksi Outlier menggunakan metode IQR.
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("no2_timeseries_interpolated.csv")

df['date'] = pd.to_datetime(df['date'])

# Hitung IQR
Q1 = df['NO2'].quantile(0.25)
Q3 = df['NO2'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter outlier
outliers_iqr = df[(df['NO2'] < lower_bound) | (df['NO2'] > upper_bound)]

print("Jumlah Outlier (IQR):", len(outliers_iqr))
print(outliers_iqr[['date', 'NO2']].head())

```
output:
```
Jumlah Outlier (IQR): 14
         date       NO2
45 2023-11-15  0.000051
46 2023-11-16  0.000044
48 2023-11-18  0.000051
67 2023-12-07  0.000047
68 2023-12-08  0.000045
```
**visualisasi outlier**
```
# === Visualisasi ===
plt.figure(figsize=(15,5))
plt.plot(df['date'], df['NO2'], label="NO2", linewidth=1)

# Titik Outlier
plt.scatter(outliers_iqr['date'], outliers_iqr['NO2'], 
            color='red', marker='o', label="Outliers")

# Garis batas atas & bawah
plt.axhline(upper_bound, color='orange', linestyle='dashed', label="Upper Bound (IQR)")
plt.axhline(lower_bound, color='blue', linestyle='dashed', label="Lower Bound (IQR)")

plt.title("Deteksi Outlier Data NO2 (Metode IQR)")
plt.xlabel("Tanggal")
plt.ylabel("Kadar NO2")
plt.legend()
plt.tight_layout()
plt.xticks(
    ticks=[df['date'].iloc[0], df['date'].iloc[-1]],
    labels=[df['date'].iloc[0].strftime('%Y-%m-%d'),
            df['date'].iloc[-1].strftime('%Y-%m-%d')]
)
plt.show()
```
![original image](https://cdn.mathpix.com/snip/images/4UfsWSRwhZQ4s_0EkoiKIOUVCyekPNegLJHxbMxH8rg.original.fullsize.png)

Setelah itu, kita akan menghapus data outlier. Karena data ini merupakan data Time Series, maka data outlier yang dihapus akan diisi kembali menggunakan Interpolasi Linear.

```
# Tandai outlier menjadi NaN
df['NO2_cleaned'] = df['NO2'].mask((df['NO2'] < lower_bound) | (df['NO2'] > upper_bound))

print("Jumlah nilai yang dinyatakan sebagai outlier:", df['NO2_cleaned'].isna().sum())

# Interpolasi linear untuk mengisi kembali nilai outlier
df['NO2_filled'] = df['NO2_cleaned'].interpolate(method='linear')

# Jika masih tersisa NaN di ujung data, isi dengan forward/backward fill
df['NO2_filled'] = df['NO2_filled'].bfill().ffill()
# df['NO2_filled'] = df['NO2_filled'].fillna(method='bfill').fillna(method='ffill')

print("Jumlah missing setelah interpolasi:", df['NO2_filled'].isna().sum())
```
Visualisasi data setelah menghapus Outlier dan mengisi kembali menggunakan Interpolasi Linear:
```
plt.figure(figsize=(15,5))
# Plot data hasil interpolasi
plt.plot(df['date'], df['NO2_filled'], label="NO2 (Interpolated)", linewidth=1)
# Tampilkan hanya tanggal awal dan akhir di sumbu X
plt.xticks(
    ticks=[df['date'].iloc[0], df['date'].iloc[-1]],
    labels=[df['date'].iloc[0].strftime('%Y-%m-%d'),
            df['date'].iloc[-1].strftime('%Y-%m-%d')]
)
plt.title("Plot Data NO2 Setelah Outlier Removal & Interpolasi")
plt.xlabel("Tanggal")
plt.ylabel("Kadar NO2")
plt.legend()
plt.tight_layout()
plt.show()
```
![original image](https://cdn.mathpix.com/snip/images/snsNmbpG5k4zT-gR6UeiISs56kbr9daDa4nXqm5ifqo.original.fullsize.png)
# 4. Modeling menggunakan KNN Regression
Dengan data Time Series kadar NO2 harian di daerah Sumenep, kita akan memprediksi kadar NO2 satu hari yang akan datang. Sekarang kita akan ubah data, mencoba mencari korelasi antara 1 hari dengan 4 hari sebelumnya. Kita juga akan membandingkan apakah semakin banyak hari sebelumnya, model akan lebih bagus?
```
import pandas as pd

def create_supervised(data, n_lag=4):
    df_supervised = pd.DataFrame()
    
    # Membuat fitur t-4 sampai t-1
    for i in range(n_lag, 0, -1):
        df_supervised[f'NO2(t-{i})'] = data.shift(i)
    
    # Label hari H
    df_supervised['NO2(t)'] = data
    
    # Hapus baris yang masih mengandung NaN akibat shift
    df_supervised.dropna(inplace=True)
    
    return df_supervised

# contoh penggunaan
supervised_df30 = create_supervised(df['NO2_filled'], n_lag=30)

# Ambil semua lag dan kolom target
lag_cols = supervised_df30.drop(columns="NO2(t)").columns
correlations = supervised_df30[lag_cols].corrwith(supervised_df30['NO2(t)'])

# Tampilkan nilai korelasi
print(correlations)
```
ouput:

```
NO2(t-30)    0.442305
NO2(t-29)    0.454453
NO2(t-28)    0.475321
NO2(t-27)    0.411415
NO2(t-26)    0.381489
NO2(t-25)    0.368746
NO2(t-24)    0.353040
NO2(t-23)    0.364877
NO2(t-22)    0.372361
NO2(t-21)    0.380395
NO2(t-20)    0.350779
NO2(t-19)    0.342422
NO2(t-18)    0.312533
NO2(t-17)    0.283250
NO2(t-16)    0.288271
NO2(t-15)    0.292097
NO2(t-14)    0.311898
NO2(t-13)    0.327056
NO2(t-12)    0.341676
NO2(t-11)    0.374011
NO2(t-10)    0.397309
NO2(t-9)     0.419212
NO2(t-8)     0.455895
NO2(t-7)     0.462417
NO2(t-6)     0.460079
NO2(t-5)     0.491438
NO2(t-4)     0.523747
NO2(t-3)     0.593804
NO2(t-2)     0.675922
NO2(t-1)     0.796428
dtype: float64

```