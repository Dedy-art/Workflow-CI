# Tugas Akhir: Predictive Maintenance - Kriteria 3 (Workflow CI)

Repositori ini berisi implementasi Workflow CI (Continuous Integration) menggunakan MLflow dan GitHub Actions untuk re-training model secara otomatis guna memenuhi kriteria penilaian Dicoding.

## Struktur Repositori
- `.github/workflows/ci.yaml` : File konfigurasi alur otomatisasi GitHub Actions.
- `MLProject/` : Folder utama proyek MLflow.
  - `MLProject` : File konfigurasi entry point MLflow.
  - `conda.yaml` : File konfigurasi environment.
  - `modelling.py` : Script utama training Machine Learning.
  - `predictive_maintenance_preprocessed.csv` : Dataset yang digunakan.

## Tautan Resmi Proyek (Submission)
- **Tautan GitHub Repository:** https://github.com/Dedy-art/Workflow-CI
- **Tautan Docker Hub (Target Advanced):** https://hub.docker.com/r/dedydesco/predictive-maintenance-model
