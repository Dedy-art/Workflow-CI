import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn

def main():
    # 1. AKTIFKAN MLFLOW AUTOLOG (Wajib sesuai standar otomatisasi pelacakan CI)
    mlflow.sklearn.autolog()

    # Memastikan path dataset mengarah dengan benar ke dalam folder MLProject
    dataset_path = "MLProject/predictive_maintenance_preprocessed.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "predictive_maintenance_preprocessed.csv"

    if not os.path.exists(dataset_path):
        print(f"Error: File {dataset_path} tidak ditemukan!")
        return

    # Load data
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=['Machine failure'], errors='ignore')
    y = df['Machine failure']
    
    # Split data training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Memulai eksekusi run via MLflow tracking standar (Tanpa run_name manual agar sinkron dengan CLI)
    with mlflow.start_run():
        print("Sedang melatih model via perintah otomatis: mlflow run...")
        
        # Inisialisasi model baseline polos tanpa tuning (Sesuai kriteria)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Prediksi dan hitung akurasi
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Training Selesai! Akurasi Model di Pipeline CI: {accuracy:.4f}")
        print("Seluruh parameter, metrik, dan model otomatis dicatat oleh MLflow Autolog.")

if __name__ == "__main__":
    main()