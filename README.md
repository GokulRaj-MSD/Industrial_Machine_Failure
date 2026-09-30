# Industrial Machine Failure Prediction

## Problem 21 — Track 1 Capstone

This project predicts whether an industrial machine will experience failure (`0 = No Failure`, `1 = Failure`) from operating conditions.

### Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository. It contains 10,000 instances and no missing values according to UCI. The dataset is synthetic but designed to reflect predictive-maintenance data encountered in industry.

UCI page: https://archive.ics.uci.edu/dataset/601/ai4i
DOI: https://doi.org/10.24432/C5HS5C

Run `python download_dataset.py` to retrieve the dataset through the `ucimlrepo` package. The downloaded file is saved as `data/ai4i2020.csv`.

### Features used

- Type
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

The identifier columns and failure-mode columns are intentionally not used as predictive inputs to reduce leakage risk.

### Models

The script compares three foundational models requested by the capstone direction:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

The final baseline is selected using F1-score because machine failure is an imbalanced binary classification problem and both false negatives and false positives matter.

### Project structure

```text
industrial_machine_failure/
├── app.py
├── machine_failure.py
├── download_dataset.py
├── requirements.txt
├── README.md
├── data/
│   └── ai4i2020.csv          # created by download_dataset.py
├── models/
│   └── machine_failure_model.pkl  # created by machine_failure.py
└── outputs/
    ├── model_comparison.csv
    ├── confusion_matrix.png
    ├── class_distribution.png
    ├── tool_wear_vs_failure.png
    └── roc_curves.png
```

## Windows PowerShell — How to run

### 1. Open the project folder

```powershell
cd "C:\path\to\industrial_machine_failure"
```

### 2. Create a virtual environment

```powershell
py -m venv .venv
```

Activate it. If PowerShell blocks `Activate.ps1`, you can skip activation and call the venv Python directly as shown below.

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install packages

If your PowerShell execution policy blocks `pip`, use `python -m pip` or the full venv path:

```powershell
python -m pip install -r requirements.txt
```

### 4. Download the dataset

```powershell
python download_dataset.py
```

You should see `Saved 10,000 rows to data/ai4i2020.csv`.

### 5. Train and evaluate

```powershell
python machine_failure.py
```

This prints data-quality checks, class distribution, model metrics and the selected model. It also creates `models/machine_failure_model.pkl` and charts in `outputs/`.

### 6. Start the Streamlit app

```powershell
python -m streamlit run app.py
```

Streamlit will display a local address, normally `http://localhost:8501`. Open it in your browser.

### If `python` is not recognized

Try:

```powershell
py -m pip install -r requirements.txt
py download_dataset.py
py machine_failure.py
py -m streamlit run app.py
```

### If PowerShell says `npm.ps1` / script execution is blocked

That issue is for Node/npm projects. This project is Python-based, so use the Python commands above; npm is not required.

## Important

Run `download_dataset.py` and `machine_failure.py` before starting Streamlit. Do not invent model scores in the report: copy the actual metrics printed by your run into the Results section.

## Capstone alignment

The project covers data quality checks, duplicates, class balance, preprocessing, foundational model comparison, classification metrics, confusion matrix, ROC-AUC, and a lightweight Streamlit prototype.
