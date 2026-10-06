import os
import re
import json
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


warnings.filterwarnings("ignore")

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# FIX (post-review): the target column was previously auto-detected and matched
# "family_diabetes" (family history) instead of the actual outcome column
# "diabetic". That let the real "diabetic" label leak into the model as an
# ordinary input feature and produced meaningless ~99% accuracy figures.
# The target is now pinned explicitly.
TARGET_COLUMN_OVERRIDE = "diabetic"

# Deployment model: chosen for screening recall, not for top ROC-AUC/accuracy.
# See paper Section 4.1 for the rationale (a high-AUC model with low recall on
# the diabetic class is unsafe for a screening tool).
DEPLOY_MODEL_NAME = "XGBoost"


def clean_column_name(column_name):
    column_name = str(column_name).strip()
    column_name = re.sub(r"[^A-Za-z0-9_]+", "_", column_name)
    column_name = re.sub(r"_+", "_", column_name)
    return column_name.strip("_")


def find_dataset_file():
    files = []

    for extension in ["*.csv", "*.xlsx", "*.xls"]:
        files.extend(DATA_DIR.glob(extension))

    if not files:
        raise FileNotFoundError(
            f"No dataset found inside: {DATA_DIR}\n"
            "Please put your diabetes dataset file inside the data folder."
        )

    print("Available dataset files:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file.name}")

    return files[0]


def load_dataset(file_path):
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    else:
        raise ValueError("Only CSV, XLSX, or XLS files are supported.")


def normalize_label(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def encode_target_value(value):
    normalized = normalize_label(value)

    non_diabetic_values = {
        "0", "no", "n", "false", "negative", "normal",
        "non_diabetic", "non_diabetes", "non_diabetic_patient",
        "not_diabetic", "non_t2dm"
    }

    diabetic_values = {
        "1", "yes", "y", "true", "positive", "diabetic",
        "diabetes", "diabetic_patient", "type_2_diabetic",
        "type2_diabetic", "t2dm"
    }

    if normalized in non_diabetic_values:
        return 0

    if normalized in diabetic_values:
        return 1

    try:
        numeric_value = float(value)
        if numeric_value == 0:
            return 0
        if numeric_value == 1:
            return 1
    except Exception:
        pass

    return np.nan


def is_binary_like(series):
    temp = series.dropna().apply(encode_target_value)
    unique_values = set(temp.dropna().unique().tolist())
    return unique_values.issubset({0, 1}) and len(unique_values) == 2


def detect_target_column(df):
    if TARGET_COLUMN_OVERRIDE is not None:
        if TARGET_COLUMN_OVERRIDE not in df.columns:
            raise ValueError(f"TARGET_COLUMN_OVERRIDE '{TARGET_COLUMN_OVERRIDE}' not found.")
        return TARGET_COLUMN_OVERRIDE

    target_keywords = ["diabetes", "diabetic", "outcome", "target", "label", "class", "status"]

    candidates = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in target_keywords):
            candidates.append(col)

    valid_candidates = [col for col in candidates if is_binary_like(df[col])]

    if valid_candidates:
        return valid_candidates[0]

    for col in df.columns:
        if is_binary_like(df[col]):
            return col

    last_col = df.columns[-1]
    print("\nTarget column could not be detected automatically.")
    print("Available columns:")
    for col in df.columns:
        print("-", col)

    raise ValueError(
        f"Please set TARGET_COLUMN_OVERRIDE manually in train_model.py.\n"
        f"Example: TARGET_COLUMN_OVERRIDE = '{last_col}'"
    )


def replace_medical_zero_values(df, target_column):
    zero_missing_keywords = [
        "glucose", "insulin", "bmi", "body_mass", "pressure",
        "bp", "skin", "thickness", "diastolic", "systolic"
    ]

    for col in df.columns:
        if col == target_column:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        col_lower = col.lower()
        if any(keyword in col_lower for keyword in zero_missing_keywords):
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                print(f"Replacing {zero_count} zero values with missing values in: {col}")
                df[col] = df[col].replace(0, np.nan)

    return df


def create_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_prediction_probability(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate_model(y_true, y_probability, threshold=0.5):
    y_pred = (y_probability >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1_Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_probability)
    }


def tune_threshold(y_true, y_probability, optimize_for="F1_Score"):
    best_threshold = 0.5
    best_score = -1
    threshold_results = []

    for threshold in np.arange(0.10, 0.91, 0.01):
        metrics = evaluate_model(y_true, y_probability, threshold)
        metrics["Threshold"] = threshold
        threshold_results.append(metrics)

        if metrics[optimize_for] > best_score:
            best_score = metrics[optimize_for]
            best_threshold = threshold

    return best_threshold, pd.DataFrame(threshold_results)


def build_feature_metadata(X_train):
    metadata = {
        "numeric_features": [],
        "categorical_features": [],
        "numeric_defaults": {},
        "categorical_defaults": {},
        "categorical_options": {}
    }

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    metadata["numeric_features"] = numeric_features
    metadata["categorical_features"] = categorical_features

    for col in numeric_features:
        median_value = X_train[col].median()
        if pd.isna(median_value):
            median_value = 0.0
        metadata["numeric_defaults"][col] = float(median_value)

    for col in categorical_features:
        mode_value = X_train[col].mode()
        default_value = str(mode_value.iloc[0]) if len(mode_value) > 0 else "Unknown"

        options = sorted(X_train[col].dropna().astype(str).unique().tolist())

        if default_value not in options:
            options.insert(0, default_value)

        metadata["categorical_defaults"][col] = default_value
        metadata["categorical_options"][col] = options

    return metadata


def main():
    print("=" * 70)
    print("Type-2 Diabetes Prediction Model Training")
    print("=" * 70)

    dataset_path = find_dataset_file()
    print(f"\nUsing dataset: {dataset_path}")

    df = load_dataset(dataset_path)
    print("\nOriginal dataset shape:", df.shape)

    df.columns = [clean_column_name(col) for col in df.columns]
    print("\nCleaned columns:")
    print(df.columns.tolist())

    target_column = detect_target_column(df)
    print(f"\nDetected target column: {target_column}")

    df[target_column] = df[target_column].apply(encode_target_value)

    before_drop = len(df)
    df = df.dropna(subset=[target_column])
    after_drop = len(df)
    if before_drop - after_drop > 0:
        print(f"Dropped {before_drop - after_drop} rows with invalid target values.")

    df[target_column] = df[target_column].astype(int)
    df = df.drop_duplicates()

    print("\nTarget distribution:")
    print(df[target_column].value_counts())

    if df[target_column].nunique() != 2:
        raise ValueError("Target column must contain exactly two classes: 0 and 1.")

    df = replace_medical_zero_values(df, target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    print("\nData split completed:")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    feature_metadata = build_feature_metadata(X_train)
    numeric_features = feature_metadata["numeric_features"]
    categorical_features = feature_metadata["categorical_features"]

    print("\nNumeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", create_one_hot_encoder())
        ])
        transformers.append(("cat", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=RANDOM_STATE
        )
        print("\nXGBoost found and added.")
    except Exception:
        print("\nXGBoost not installed. Continuing without XGBoost.")

    class_counts = y_train.value_counts()
    minority_count = class_counts.min()
    use_smote = minority_count > 5

    if use_smote:
        smote_k = min(5, minority_count - 1)
        print(f"\nSMOTE enabled with k_neighbors={smote_k}")
    else:
        print("\nSMOTE disabled because minority class is too small.")

    trained_models = {}
    validation_results = []

    for model_name, classifier in models.items():
        print(f"\nTraining model: {model_name}")

        steps = [("preprocessor", preprocessor)]
        if use_smote:
            steps.append(("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=smote_k)))
        steps.append(("classifier", classifier))

        pipeline = ImbPipeline(steps=steps)
        pipeline.fit(X_train, y_train)

        y_val_probability = get_prediction_probability(pipeline, X_val)
        metrics = evaluate_model(y_val, y_val_probability, threshold=0.5)
        metrics["Model"] = model_name

        validation_results.append(metrics)
        trained_models[model_name] = pipeline

        print(classification_report(y_val, (y_val_probability >= 0.5).astype(int), zero_division=0))

    validation_results_df = pd.DataFrame(validation_results)
    validation_results_df = validation_results_df[
        ["Model", "Accuracy", "Precision", "Recall", "Specificity", "F1_Score", "ROC_AUC"]
    ]
    validation_results_df = validation_results_df.sort_values(by="F1_Score", ascending=False)

    print("\nValidation model comparison (for paper reporting):")
    print(validation_results_df)
    validation_results_df.to_csv(OUTPUT_DIR / "validation_model_comparison.csv", index=False)

    # --- Deployment model selection -------------------------------------
    # The table above ranks by F1 on the validation set, for the paper.
    # The DEPLOYED model is fixed to DEPLOY_MODEL_NAME (default: XGBoost)
    # because it has materially higher recall on the diabetic class, which
    # matters more than raw accuracy/ROC-AUC for a screening tool. See
    # paper Section 4.1.
    if DEPLOY_MODEL_NAME in trained_models:
        best_model_name = DEPLOY_MODEL_NAME
        print(f"\nDeployment model fixed to: {best_model_name} (recall-first policy)")
    else:
        best_model_name = validation_results_df.iloc[0]["Model"]
        print(f"\n{DEPLOY_MODEL_NAME} unavailable, falling back to top validation model: {best_model_name}")

    best_model = trained_models[best_model_name]

    y_val_probability = get_prediction_probability(best_model, X_val)
    best_threshold, threshold_results_df = tune_threshold(y_val, y_val_probability, optimize_for="F1_Score")
    print(f"Best F1 threshold for deployed model: {best_threshold:.2f}")

    threshold_results_df.to_csv(OUTPUT_DIR / "threshold_tuning_results.csv", index=False)

    y_test_probability = get_prediction_probability(best_model, X_test)
    test_metrics = evaluate_model(y_test, y_test_probability, threshold=best_threshold)
    y_test_pred = (y_test_probability >= best_threshold).astype(int)

    print("\nFinal test results (deployed model):")
    print(test_metrics)
    print("\nFinal test classification report:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    test_results_df = pd.DataFrame([test_metrics])
    test_results_df["Best_Model"] = best_model_name
    test_results_df["Best_Threshold"] = best_threshold
    test_results_df.to_csv(OUTPUT_DIR / "final_test_results.csv", index=False)

    feature_columns = X.columns.tolist()

    joblib.dump(best_model, MODEL_DIR / "diabetes_prediction_model.pkl")
    joblib.dump(best_threshold, MODEL_DIR / "best_threshold.pkl")
    joblib.dump(feature_columns, MODEL_DIR / "feature_columns.pkl")
    joblib.dump(feature_metadata, MODEL_DIR / "feature_metadata.pkl")

    project_info = {
        "dataset_file": dataset_path.name,
        "target_column": target_column,
        "deployed_model": best_model_name,
        "deployed_threshold": float(best_threshold),
        "train_size": int(X_train.shape[0]),
        "validation_size": int(X_val.shape[0]),
        "test_size": int(X_test.shape[0]),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()}
    }

    with open(OUTPUT_DIR / "project_info.json", "w") as file:
        json.dump(project_info, file, indent=4)

    print("\nModel files saved successfully:")
    print(MODEL_DIR / "diabetes_prediction_model.pkl")
    print(MODEL_DIR / "best_threshold.pkl")
    print(MODEL_DIR / "feature_columns.pkl")
    print(MODEL_DIR / "feature_metadata.pkl")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
