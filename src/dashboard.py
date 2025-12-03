import streamlit as st
import mlflow
import pandas as pd
import os

st.title("Student Performance Dashboard")

# MLflow tracking URI - gunakan file-based langsung
tracking_uri = "file:///app/mlruns"
mlflow.set_tracking_uri(tracking_uri)

st.write(f"**MLflow Tracking URI:** {tracking_uri}")

# Get latest experiment
client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

try:
    # Ambil semua runs dari experiment default (ID 0)
    runs = client.search_runs(experiment_ids=["0"], order_by=["start_time DESC"], max_results=10)
    
    if runs:
        st.write(f"**Total Runs:** {len(runs)}")
        
        # Tampilkan run terbaik
        best_run = runs[0]
        st.subheader("Model Terbaik")
        
        if "accuracy" in best_run.data.metrics:
            st.metric("Akurasi", f"{best_run.data.metrics['accuracy']:.4f}")
        
        st.write("**Parameter:**")
        for key, value in best_run.data.params.items():
            st.write(f"- {key}: {value}")
        
        st.write(f"**Run ID:** {best_run.info.run_id}")
        st.write(f"**Start Time:** {pd.to_datetime(best_run.info.start_time, unit='ms')}")
        
        # Tampilkan tabel semua runs yang valid (ada metrik accuracy)
        st.subheader("Riwayat Eksperimen")
        data = []
        for run in runs:
            # Hanya tampilkan run yang memiliki metrik accuracy
            if "accuracy" in run.data.metrics:
                data.append({
                    "Run ID": run.info.run_id[:8],
                    "Accuracy": f"{run.data.metrics['accuracy']:.4f}",
                    "Parameters": ", ".join([f"{k}={v}" for k, v in run.data.params.items()]),
                    "Duration": f"{(run.info.end_time - run.info.start_time) / 1000:.1f}s" if run.info.end_time else "Running"
                })
        
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Belum ada run dengan metrik lengkap.")
    else:
        st.warning("Belum ada run MLflow. Jalankan pipeline training terlebih dahulu.")
        st.info("Jalankan: `docker exec mlflow python src/train.py`")
        
except Exception as e:
    st.error(f"Error mengakses MLflow: {str(e)}")
    st.info(f"Pastikan MLflow server berjalan dan shared volume ter-mount dengan benar")
