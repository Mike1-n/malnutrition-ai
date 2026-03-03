# Child Malnutrition Clinical Assessment App & AI Support System

This application is a Streamlit-based clinical dashboard designed to predict malnutrition risk in children under 5 using both standard clinical indicators (like WHZ) and a Machine Learning model trained on comprehensive longitudinal patient data (including medical history, feeding practices, and socio-economic factors).

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
    -   Analyzes **20+ multi-dimensional features**, including growth trends, clinical history (e.g., HIV, Immunization), feeding practices, and socio-economic status.
    -   *Why it's needed?* It can detect a child "At Risk" closer to a high probability if they have stacked risk factors (e.g., poor sanitation, underlying chronic illness, and lack of immunization), even if their single current physical measurement is near normal, allowing for **early intervention**.

2.  **Standard Clinical Rules**:
    -   Uses strict **WHO Weight-for-Height Z-Score (WHZ) cutoffs**.
    -   *Why it's needed?* To ensure no child meeting standard "Severe" criteria is ever missed, regardless of the AI prediction.

---

## 🧠 Machine Learning Logic

### 1. Synthetic Data Generation (`generate_synthetic_data.py`)
Since real medical records were not provided, this script simulates realistic patient histories:
-   **Method**: Generates virtual children, each with multiple visits (approx. 1 month apart).
-   **Comprehensive Profiles**: Simulates socio-economic status (income, education, water access) and medical background (HIV exposure, congenital diseases, immunization tracking).
-   **Target Variable (`malnutrition`)**:
    -   **1 (Malnourished)** if: Synthetic Z-score proxy drops below -2, or if cumulative risk factors (like zero-dose immunization, HIV infection, poor feeding) push the vulnerability threshold.
    -   **0 (Healthy)** otherwise.
-   **Noise**: Random label flipping to simulate real-world uncertainty.

### 2. Model Training (`malnutrition_model.py`)
The model uses a **Longitudinal Approach**, meaning it looks at *trends* and *context* rather than just a single physical snapshot.
-   **Algorithm**: Random Forest Classifier (100 trees).
-   **Feature Engineering (21 Features Used)**:
    -   **Growth & Trends**: `weight`, `height`, `WHZ`, `weight_change` (difference from previous visit), `whz_change`.
    -   **Demographics**: `gender`, `birth_weight`.
    -   **Medical History**: `illness_count_roll`, `illness_count_last_month`, `immunization_status`, `hiv_exposure`, `chronic_illness`, `congenital_disease`, `recurrent_diarrhea`.
    -   **Feeding Practices**: `exclusive_breastfeeding_6m`, `feeding_diversity_score`, `meal_frequency_per_day`.
    -   **Socio-Economic & WASH**: `household_income_level`, `parent_education_level`, `access_to_clean_water`, `sanitation_access`.
-   **Performance**: The model is evaluated using Accuracy, Precision, Recall, and ROC-AUC. Graphs are saved in the `results/` folder.

---

## 📊 Clinical Calculations

The app computes several standard metrics to assist clinicians:

### 1. Weight-for-Height Z-Score (WHZ)
This is the gold standard primary indicator for identifying wasting.
-   **Method**: WHO LMS (Lambda-Mu-Sigma) Method.
-   **Data Source**: Uses `who_standards.csv` lookup table (Gender/Height).
-   **Formula**:
    $Z = \frac{(Weight / M)^L - 1}{L \times S}$
-   **Interpretation**:
    -   **< -3 SD**: Severe Acute Malnutrition (SAM)
    -   **-3 to -2 SD**: Moderate Acute Malnutrition (MAM)
    -   **> -2 SD**: Normal

### 2. Criteria for Malnutrition Diagnosis

The application flags a child as **At Risk** or requires **Critical Action** based on a combination of factors:

| Indicator / Condition | Critical Finding (Red/Severe) | Moderate Risk (Orange/Warning) | Normal / Low Risk (Green) |
| :--- | :--- | :--- | :--- |
| **WHZ Score** | < -3 SD (Severe Wasting) | -3 to -2 SD (Moderate Wasting) | > -2 SD |
| **ML Prediction** | Probability ≥ 70% | Probability 30% – 69% | Probability < 30% |
| **Immunization** | Zero Dose | Partially Immunized | Age Appropriate / Fully |
| **HIV Status** | HIV Infected | HIV Exposed Unaffected | HIV Unexposed |

> **Note**: The **Machine Learning Model** and standard clinical alerts check multiple dimensions. For instance, an early stop to exclusive breastfeeding (< 2 months) or a "Low" Socio-Economic Status score (< 5/13) will also trigger clinical warnings.

---

## 🖥️ App Workflow (`app.py`)

1.  **Input**: The clinician enters comprehensive data including longitudinal measurements (last 5 visits), health history, feeding practices, and household assets.
2.  **Processing**:
    -   Calculates current WHZ and trends (weight change, WHZ change).
    -   Computes an aggregate Socio-Economic Status (SES) score out of 13 based on education, occupation, household crowding, and assets.
3.  **Prediction**:
    -   The engineered 21-feature vector is fed into the loaded `malnutrition_model.pkl`.
    -   The model outputs a probability (0-100%) of malnutrition risk based on holistic patterns.
4.  **Visualization**:
    -   **Trend Trajectories**: High-visibility line charts showing Weight and WHZ trends over time.
    -   **Risk Gauges**: Visual indicators for ML Prediction and current WHZ category.
5.  **Recommendations**: Logic-based advice is dynamically generated based on the highest risk factors (e.g., flagging immediate vaccination referral for "Zero Dose" children, or clinical management for HIV+ status).
