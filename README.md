# Sistem Prediksi Kurs Mata Uang

Aplikasi web prediksi kurs menggunakan Prophet + Streamlit.

## Struktur Folder

```
kurs_app/
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── prophet_USD.pkl
    ├── prophet_SGD.pkl
    ├── prophet_MYR.pkl
    ├── prophet_JPY.pkl
    ├── prophet_CNY.pkl
    ├── prophet_EUR.pkl
    ├── prophet_AUD.pkl
    └── prophet_SAR.pkl
```

## Cara Pakai

### 1. Siapkan model
Download semua file `.pkl` dari Google Drive (`Skripsi/Models/`) ke folder `models/`.

### 2. Install dependency
```bash
pip install -r requirements.txt
```

### 3. Jalankan
```bash
streamlit run app.py
```

## Fitur
- Pilih mata uang (USD, SGD, MYR, JPY, CNY, EUR, AUD, SAR)
- Tabel MAPE semua mata uang
- Kurs realtime dari ExchangeRate API
- Prediksi 7 hari ke depan (tabel + grafik)
