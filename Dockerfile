# Gunakan image python
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Salin requirements dan install
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

# Salin semua kode
COPY . .

# Expose port untuk Streamlit dan MLflow
EXPOSE 8501 5000

# Default command (bisa diubah di docker-compose)
CMD ["bash"]
