# ============================================================
# INDUSTRIAL MACHINE FAILURE PREDICTION
# Problem 21 - ML Capstone
# ============================================================

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

warnings.filterwarnings("ignore")


# ============================================================
# FOLDERS
# ============================================================

DATA_PATH = "data/ai4i2020.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\n=== INDUSTRIAL MACHINE FAILURE PREDICTION ===")

df = pd.read_csv(DATA_PATH)

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# DATA QUALITY
# ============================================================

print("\nMissing values:")

if df.isnull().sum().sum() == 0:
    print("No missing values")
else:
    print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

TARGET = "Machine failure"

print("\nTarget distribution:")
print(df[TARGET].value_counts())

print("\nTarget percentages:")
print(
    df[TARGET]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# CLASS DISTRIBUTION GRAPH
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x=TARGET
)

plt.title("Machine Failure Class Distribution")
plt.xlabel("Machine Failure")
plt.ylabel("Number of Records")

plt.xticks(
    [0, 1],
    ["No Failure", "Failure"]
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "class_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear"
]

X = df[FEATURES].copy()
y = df[TARGET].copy()


print("\nFeatures used:")

for feature in FEATURES:
    print("-", feature)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))


# ============================================================
# PREPROCESSING
# ============================================================

numeric_features = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear"
]

categorical_features = [
    "Type"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=7
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42
        )
}


# ============================================================
# TRAIN MODELS
# ============================================================

results = []
trained_models = {}

print("\n=== MODEL TRAINING ===")


for model_name, model in models.items():

    print("\n----------------------------------------")
    print("Training:", model_name)
    print("----------------------------------------")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[model_name] = pipeline

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "No Failure",
                "Failure"
            ],
            zero_division=0
        )
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n=== MODEL COMPARISON ===")

print(
    results_df.to_string(
        index=False
    )
)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

best_model = trained_models[
    best_model_name
]

print("\n=== FINAL MODEL ===")

print(
    "Selected model:",
    best_model_name
)


# ============================================================
# FINAL PREDICTIONS
# ============================================================

final_predictions = best_model.predict(
    X_test
)

final_probabilities = best_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    final_predictions
)

print("\n=== CONFUSION MATRIX ===")

print(cm)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "No Failure",
        "Failure"
    ],
    yticklabels=[
        "No Failure",
        "Failure"
    ]
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# ROC CURVES
# ============================================================

plt.figure(figsize=(8, 6))

for model_name, pipeline in trained_models.items():

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC={auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.title("ROC Curve Comparison")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_curves.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# TOOL WEAR ANALYSIS
# ============================================================

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x=TARGET,
    y="Tool wear"
)

plt.title(
    "Tool Wear vs Machine Failure"
)

plt.xlabel("Machine Failure")

plt.ylabel("Tool Wear")

plt.xticks(
    [0, 1],
    ["No Failure", "Failure"]
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "tool_wear_vs_failure.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "machine_failure_model.pkl"
)

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nModel saved to:")

print(
    MODEL_PATH
)


# ============================================================
# SAMPLE PREDICTION
# ============================================================

sample_machine = pd.DataFrame({

    "Type": ["M"],

    "Air temperature": [300.0],

    "Process temperature": [310.0],

    "Rotational speed": [1500],

    "Torque": [55.0],

    "Tool wear": [200]

})


sample_prediction = best_model.predict(
    sample_machine
)[0]

sample_probability = best_model.predict_proba(
    sample_machine
)[0][1]


print("\n=== SAMPLE PREDICTION ===")

print(
    sample_machine.to_string(
        index=False
    )
)

if sample_prediction == 1:

    print(
        "\nPrediction: MACHINE FAILURE LIKELY"
    )

else:

    print(
        "\nPrediction: NO MACHINE FAILURE PREDICTED"
    )

print(
    f"Failure Probability: {sample_probability:.2%}"
)


# ============================================================
# COMPLETED
# ============================================================

print("\n========================================")
print("PROJECT TRAINING COMPLETED SUCCESSFULLY")
print("========================================")

print("\nGenerated files:")

print(
    "models/machine_failure_model.pkl"
)

print(
    "outputs/model_comparison.csv"
)

print(
    "outputs/class_distribution.png"
)

print(
    "outputs/confusion_matrix.png"
)

print(
    "outputs/roc_curves.png"
)

print(
    "outputs/tool_wear_vs_failure.png"
)

print("\nNext command:")

print(
    ".\\.venv\\Scripts\\python.exe -m streamlit run app.py"
)