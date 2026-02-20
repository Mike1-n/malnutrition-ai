# Child Malnutrition Clinical Assessment App

This application is a Streamlit-based clinical dashboard designed to predict malnutrition risk in children under 5 using both standard clinical indicators and a Machine Learning model trained on longitudinal patient data.

## 🚀 How to Run
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Generate Data (Optional)**:
    If `malnutrition_data.csv` is missing, generate synthetic data:
    ```bash
    python generate_synthetic_data.py
    ```
3.  **Train Model**:
    Train the Random Forest model on the data:
    ```bash
    python malnutrition_model.py
    ```
4.  **Launch App**:
    ```bash
    streamlit run app.py
    ```

### 🔍 Hybrid System: AI + Clinical Rules

**Yes, this system uses Artificial Intelligence.** It is a **Hybrid Support System** that combines:

1.  **Artificial Intelligence (AI)**:
    -   Uses a **Random Forest Machine Learning Model**.
    -   Analyzes **longitudinal trends** (past 5 visits) to predict risk probabilities.
    -   *Why it's needed?* It can detect a child "At Risk" closer to 70% probability if they are rapidly losing weight, even if their current MUAC is still "Green" (Normal), allowing for **early intervention**.

2.  **Standard Clinical Rules**:
    -   Uses strict **WHO/Ministry of Health cutoffs** (e.g., MUAC < 11.5cm).
    -   *Why it's needed?* To ensure no child meeting standard "Severe" criteria is ever missed, regardless of the AI prediction.

---

## 🧠 Machine Learning Logic

### 1. Synthetic Data Generation (`generate_synthetic_data.py`)
Since real medical records were not provided, this script simulates realistic patient histories:
-   **Method**: Generates 200 virtual children, each with 6 visits (1 month apart).
-   **Growth Simulation**: Linear growth in weight/height with random noise.
-   **Illness Simulation**: Randomly assigns illness (20% chance). If ill, weight and MUAC drop slightly.
-   **Target Variable (`malnutrition`)**:
    -   **1 (Malnourished)** if: MUAC < 12.5cm OR Weight-for-Height ratio is very low (< 13).
    -   **0 (Healthy)** otherwise.
    -   **Noise**: 5% random label flipping to simulate real-world uncertainty.

### 2. Model Training (`malnutrition_model.py`)
The model uses a **Longitudinal Approach**, meaning it looks at *trends* rather than just a single snapshot.
-   **Algorithm**: Random Forest Classifier (100 trees).
-   **Feature Engineering**:
    -   `weight`: Current weight (kg).
    -   `height`: Current height (cm).
    -   `MUAC`: Current Mid-Upper Arm Circumference (cm).
    -   `weight_change`: Difference in weight from the previous visit.
    -   `muac_change`: Difference in MUAC from the previous visit.
    -   `illness_count_roll`: Total number of illnesses recorded in the last 5 visits.
-   **Performance**: The model is evaluated using Accuracy, Precision, Recall, and ROC-AUC. Graphs are saved in the `results/` folder.

---

## 📊 Clinical Calculations

The app computes several standard metrics to assist clinicians:

### 1. Body Mass Index (BMI) Proxy
A simple indicator of body mass relative to height.
-   **Formula**: $BMI = \frac{Weight (kg)}{Height (m)^2}$
-   *Note*: For children, BMI z-scores are preferred, but raw BMI is provided for reference.

### 2. MUAC Classification
Mid-Upper Arm Circumference is a key indicator for mortality risk.
-   **< 11.5 cm**: Severe Acute Malnutrition (Red)
-   **11.5 - 12.4 cm**: Moderate Acute Malnutrition (Orange)
-   **12.5 - 13.5 cm**: At Risk (Yellow/Orange)
-   **≥ 13.5 cm**: Normal (Green)

### 3. Weight-for-Height Z-Score (WHZ)
This is the gold standard for identifying wasting.
-   **Method**: WHO LMS (Lambda-Mu-Sigma) Method.
-   **Data Source**: Uses a simplified `who_standards.csv` lookup table (Gender/Height).
-   **Formula**:
    $Z = \frac{(Weight / M)^L - 1}{L \times S}$
    -   *L*: Power (box-cox)
    -   *M*: Median
    -   *S*: Coefficient of variation
-   **Interpretation**:
    -   **< -3 SD**: Severe Acute Malnutrition (SAM)
    -   **-3 to -2 SD**: Moderate Acute Malnutrition (MAM)
    -   **-2 to +1 SD**: Normal/At Risk (depending on context)
    -   **> +1 SD**: Possible Overweight

### 4. Criteria for Malnutrition Diagnosis

The application flags a child as **Malnourished** if *any* of the following conditions are met:

| Indicator | Severe Acute Malnutrition (Red) | Moderate Acute Malnutrition (Orange) | At Risk (Yellow) |
| :--- | :--- | :--- | :--- |
| **MUAC** | < 11.5 cm | 11.5 – 12.4 cm | 12.5 – 13.5 cm |
| **WHZ Score** | < -3 SD | -3 to -2 SD | -2 to +1 SD |
| **ML Prediction** | Probability > 70% | Probability 30% – 70% | Probability 10% - 30% |

> **Note**: The **Machine Learning Model** may flag a child as "High Risk" even if current measurements are normal, if it detects a **rapid negative trend** (e.g., significant weight loss over the last 3 visits).

---

## 🖥️ App Workflow (`app.py`)

1.  **Input**: The clinician enters data for the **last 5 visits** in a data editor table.
2.  **Processing**:
    -   The app sorts the data by date.
    -   It computes the change in weight (`current - previous`) and sums the illness count.
3.  **Prediction**:
    -   These derived features are fed into the loaded `malnutrition_model.pkl`.
    -   The model outputs a probability (0-100%) of malnutrition risk.
4.  **Visualization**:
    -   **Weight Trajectory**: Line chart showing weight change over time.
    -   **BMI Trajectory**: Line chart showing BMI trends over time.
    -   **WHZ Status**: Calculated Z-score for the most recent visit.
5.  **Recommendations**: Logic-based advice is displayed based on the highest risk factor (Machine Learning High Risk OR Clinical Severe MUAC).
