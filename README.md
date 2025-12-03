# Student Performance Analysis MLOps Project

Proyek MLOps lengkap untuk analisis performa siswa dengan pipeline otomatis, tracking eksperimen, dan dashboard interaktif.

## 🚀 Fitur
- ✅ **MLflow** untuk tracking eksperimen dan model versioning
- ✅ **DVC** untuk data versioning dan pipeline management
- ✅ **MinIO** sebagai object storage (S3-compatible)
- ✅ **Docker & Docker Compose** untuk reproducible environment
- ✅ **GitHub Actions** untuk CI/CD pipeline
- ✅ **Streamlit** dashboard untuk visualisasi hasil eksperimen
- ✅ **Jupyter Notebook** untuk eksplorasi data

## 📁 Struktur Proyek
```
├── .github/workflows/    # CI/CD configuration
├── data/                 # Data mentah dan hasil processing
├── notebooks/            # Jupyter notebooks untuk eksplorasi
├── src/                  # Source code
│   ├── prepare.py        # Data preparation
│   ├── train.py          # Model training
│   ├── eval.py           # Model evaluation
│   └── dashboard.py      # Streamlit dashboard
├── dvc.yaml              # DVC pipeline definition
├── docker-compose.yml    # Docker services configuration
├── Dockerfile            # Docker image definition
└── requirements.txt      # Python dependencies
```

## 🏃 Quick Start

### 1. Clone & Setup
```bash
git clone <repository-url>
cd "Student Analysis DVC"
pip install -r requirements.txt
```

### 2. Jalankan dengan Docker (Recommended)
```bash
# Build dan jalankan semua services
docker-compose up --build -d

# Akses services:
# - MLflow UI: http://localhost:5000
# - Streamlit Dashboard: http://localhost:8501
# - MinIO Console: http://localhost:9001 (user: minio, pass: minio123)
```

### 3. Jalankan Pipeline DVC
```bash
# Jalankan pipeline lengkap (prepare → train → eval)
dvc repro

# Atau jalankan training di dalam container:
docker exec mlflow python src/train.py
```

### 4. Eksplorasi Data
```bash
# Buka notebook di notebooks/eksplorasi.ipynb
jupyter notebook notebooks/eksplorasi.ipynb
```

## 📊 Pipeline MLOps

### Pipeline Stages
1. **prepare**: Preprocessing data dan encoding fitur kategorikal
2. **train**: Training model RandomForest dan logging ke MLflow
3. **eval**: Evaluasi model dan export metrics

### Menjalankan Pipeline
```bash
# Full pipeline
dvc repro

# Specific stage
dvc repro prepare
dvc repro train
```

## 🎯 Hasil Eksperimen
- Model: RandomForestClassifier
- Akurasi: 100% (pada dataset demo)
- Tracking: Semua eksperimen ter-log di MLflow
- Visualisasi: Dashboard Streamlit menampilkan metrik dan riwayat eksperimen

## 🔧 Konfigurasi

### DVC Remote (MinIO)
```bash
dvc remote add -d minio_remote s3://mlops-artifact/
dvc remote modify minio_remote endpointurl http://localhost:9000
dvc remote modify minio_remote access_key_id minio
dvc remote modify minio_remote secret_access_key minio123
```

### MLflow Tracking
- Local runs: `file:///app/mlruns` (di container)
- Server: http://localhost:5000

## 🛠️ Development

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Tests (via CI/CD)
```bash
# CI/CD akan otomatis run saat push ke main branch
git push origin main
```

## 📝 Lisensi
MIT

## 👥 Kontributor
- Proyek MLOps untuk analisis performa siswa
