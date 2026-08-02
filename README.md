# Sistem Prediksi Kurs Mata Uang

Dashboard berbasis web untuk memantau nilai tukar mata uang secara real-time dan melakukan prediksi kurs terhadap Rupiah menggunakan Meta Prophet dan Streamlit.

## Fitur

- Monitoring kurs real-time menggunakan ExchangeRate API
- Visualisasi data historis
- Prediksi kurs 7 hari ke depan menggunakan Meta Prophet
- Konversi mata uang
- Mendukung mata uang USD, EUR, JPY, SGD, CNY, MYR, AUD, dan SAR

## Teknologi

- Python
- Streamlit
- Meta Prophet
- Pandas
- Plotly
- ExchangeRate API

## Struktur Folder

```
sistem-prediksi-kurs/
├── actual_data/
├── models/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

## Dataset

- Data historis kurs dari Bank Indonesia
- Data kurs real-time dari ExchangeRate API

## Screenshot


