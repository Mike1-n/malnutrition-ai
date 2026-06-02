"""
malnutrition_model.py
---------------------
Trains a Random Forest classifier on the Kenyan simulated malnutrition dataset.
Target: multi-class 'malnutrition_category' (5 classes).
Features include MUAC (mm) alongside anthropometric, clinical, and socio-economic variables.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 12


def load_data(filepath):
    """Loads the dataset from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)


def create_pipeline(numeric_features, categorical_features):
    """Creates a preprocessing + Random Forest pipeline."""

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=4,
            random_state=42,
            class_weight='balanced'   # handles any slight class imbalance
        ))
    ])

    return pipeline


def evaluate_model(model, X_test, y_test, output_dir):
    """Evaluates the model and saves plots."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n── Model Evaluation ──────────────────────────────")
    print(f"Accuracy : {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    classes = model.classes_
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix\n(Kenyan Dataset – Multi-class)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    print(f"Confusion matrix saved → {output_dir}/confusion_matrix.png")


def main():
    data_file  = 'kenyan_malnutrition_data.csv'
    model_file = 'malnutrition_model.pkl'
    results_dir = 'results'

    # 1. Load Data
    print(f"Loading data from {data_file} ...")
    try:
        df = load_data(data_file)
    except FileNotFoundError as e:
        print(e)
        print("Run prepare_kenyan_data.py first to generate the dataset.")
        return

    print(f"  Rows: {len(df)}  |  Columns: {df.columns.tolist()}")

    target = 'malnutrition_category'

    # 2. Drop columns not used as model features
    drop_cols = ['child_id', target]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[target]

    print(f"\nTarget distribution:\n{y.value_counts().to_string()}")

    # 3. Identify feature types automatically
    numeric_features     = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"\nNumeric features  : {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # 4. Train / Test split – stratified to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

    # 5. Train
    print("\nTraining Random Forest (multi-class, 200 trees) ...")
    pipeline = create_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # 6. Evaluate
    evaluate_model(pipeline, X_test, y_test, results_dir)

    # 7. Save model
    joblib.dump(pipeline, model_file)
    print(f"\nModel saved → {model_file}")
    print("Classes:", pipeline.classes_.tolist())


if __name__ == "__main__":
    main()
