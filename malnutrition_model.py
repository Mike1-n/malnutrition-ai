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
    roc_auc_score, 
    roc_curve
)

# Set plot style directly using matplotlib parameters for better compatibility
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 12

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def engineer_longitudinal_features(df):
    """
    Engineers features based on longitudinal data (past visits).
    Assumes df has 'child_id', 'visit_date', 'weight', 'height', 'WHZ', 'illness'.
    """
    df = df.sort_values(by=['child_id', 'visit_date'])
    
    # Lag Features (Change from previous visit)
    df['weight_prev'] = df.groupby('child_id')['weight'].shift(1)
    df['whz_prev'] = df.groupby('child_id')['WHZ'].shift(1)
    
    df['weight_change'] = df['weight'] - df['weight_prev']
    df['whz_change'] = df['WHZ'] - df['whz_prev']
    
    # Rolling Illness Count (Last 5 visits including current)
    # Convert illness to numeric (1 if yes, 0 if no) for rolling sum
    df['illness_num'] = df['illness'].apply(lambda x: 1 if x == 'yes' else 0)
    df['illness_count_roll'] = df.groupby('child_id')['illness_num'].transform(lambda x: x.rolling(window=5, min_periods=1).sum())
    
    # Fill NA for first visits (no history)
    df['weight_change'] = df['weight_change'].fillna(0)
    df['whz_change'] = df['whz_change'].fillna(0)
    
    return df

def create_pipeline(numeric_features, categorical_features):
    """Creates a preprocessing and modeling pipeline."""
    
    # Preprocessing for numeric data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Random Forest Classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    return pipeline

def evaluate_model(model, X_test, y_test, output_dir):
    """Evaluates the model and saves plots."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC: {roc_auc:.4f}")
    except ValueError:
        print("ROC-AUC could not be calculated.")
        roc_auc = None
    
    print(f"\nModel Evaluation Metrics:")
    print(f"Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # ROC Curve
    if roc_auc is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()

def main():
    data_file = 'malnutrition_data.csv'
    model_file = 'malnutrition_model.pkl'
    results_dir = 'results'

    # 1. Load Data
    print(f"Loading data from {data_file}...")
    try:
        df = load_data(data_file)
    except FileNotFoundError:
        print("Dataset not found.")
        return

    # 2. Feature Engineering (Longitudinal)
    print("Engineering longitudinal features...")
    df = engineer_longitudinal_features(df)

    target = 'malnutrition'
    
    # Drop identifiers and intermediate columns
    # We KEEP features for prediction: 
    # Current: weight, height, WHZ, gender, birth_weight, + ALL NEW SOCIO/MED Features.
    
    # List of columns to EXCLUDE from training (Target is 'malnutrition')
    features_to_drop = ['child_id', 'visit_date', 'weight_prev', 'whz_prev', 'illness', 'illness_num', 'food_types', target]
    
    features_to_drop = [col for col in features_to_drop if col in df.columns]

    X = df.drop(columns=features_to_drop)
    y = df[target]

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Features used for training: {numeric_features + categorical_features}")

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # 4. Train Model
    print("Training Random Forest Classifier (Longitudinal)...")
    model = create_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)

    # 5. Evaluate Model
    evaluate_model(model, X_test, y_test, results_dir)

    # 6. Save Model
    joblib.dump(model, model_file)
    print(f"\nModel saved to {model_file}")

if __name__ == "__main__":
    main()
