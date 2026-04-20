# Flight Delay Analysis and Prediction

**Course:** PAML Spring 2026  
**Team:** Lin Wu, Yufan Peng, Xinrui Song, Xinyi Huang, Zhiying Huo

---

## Project Overview

This project analyzes U.S. domestic flight delay patterns using BTS On-Time Performance data (2025). It includes:

- Exploratory Data Analysis (EDA) with visualizations
- Random Forest classification — predict whether a flight will be delayed ≥ 15 min
- Random Forest regression — predict the duration of a delay
- An interactive Streamlit web app for exploring airline/route performance

---

## Repository Structure

```
Final_project_code/
├── app.py                      # Streamlit web application
├── flight_delay_colab.ipynb    # Full ML pipeline notebook
├── requirements.txt            # Python dependencies
├── data/                       # Place your CSV dataset files here (not tracked by git)
├── scripts/
│   └── sample_code.py          # Reference script
└── outputs/                    # Auto-generated plots and metrics
```

---

## Dataset

Data source: [BTS Airline On-Time Performance](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGK&QO_fu146_anzr=b0-gvzr)

Download the monthly **On-Time Reporting** CSV files and place them in your local dataset folder (e.g. `~/Downloads` or the `data/` folder in this repo).

Files used in this project:
- `202501T_ONTIME_REPORTING.csv`
- `202502T_ONTIME_REPORTING.csv`
- `202503T_ONTIME_REPORTING.csv`
- `202505T_ONTIME_REPORTING.csv`
- `202508T_ONTIME_REPORTING.csv`
- `202509T_ONTIME_REPORTING.csv`
- `202511T_ONTIME_REPORTING.csv`
- `202512T_ONTIME_REPORTING.csv`

---

## Setup

### 1. Install dependencies

```bash
pip3 install -r requirements.txt
```

### 2. Set your dataset path

Open `flight_delay_colab.ipynb` and update **Cell 1 (Configuration)**:

```python
data_directory = '/your/path/to/csv/files'
```

Open `app.py` and update line 24:

```python
DATA_DIR = "/your/path/to/csv/files"
```

---

## Running the Notebook

Open `flight_delay_colab.ipynb` in Jupyter or VS Code and run all cells in order.

```bash
jupyter notebook flight_delay_colab.ipynb
```

---

## Running the Streamlit App

```bash
cd Final_project_code
python3 -m streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### App Features

| Tab | Description |
|-----|-------------|
| Route Explorer | Search origin → destination, view delay stats and model predictions |
| Airline Comparison | Compare airlines by delay probability, expected delay, and reliability |
| Delay Factors | Visualize breakdown of delay causes (weather, carrier, NAS, etc.) |

---

## Model Performance

Results are saved in `outputs/` after running the notebook:

| Model | Task | Key Metric |
|-------|------|------------|
| Random Forest Classifier | Delay ≥ 15 min (Yes/No) | F1-score (see `classification_metrics.json`) |
| Random Forest Regressor | Delay duration (minutes) | RMSE (see `regression_metrics.json`) |
