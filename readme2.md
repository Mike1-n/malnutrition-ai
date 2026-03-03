# Data Analysis Plan

Data analysis will be conducted using the Python programming language in a standard Python environment, utilizing relevant libraries for data manipulation, modeling, and visualization, including `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, and `NumPy`.

The plan includes data cleaning, feature preparation, model training, performance evaluation, software & reporting, and data security & confidentiality.

## 1. Data Cleaning
- All collected longitudinal patient data will first be checked for completeness and consistency.
- **Handling Missing Data:**
   - Missing continuous numeric values will be imputed using the **median** (via `SimpleImputer`).
   - Missing categorical variables will be filled with a constant 'missing' value.
- **Data Scaling & Encoding:**
   - **Continuous Variables** (e.g., weight, height, WHZ score, weight change, WHZ change): Analyzed directly and normalized using **StandardScaler** to ensure uniform model training.
   - **Categorical Variables** (e.g., gender, immunization status, HIV exposure, socio-economic status, feeding practices): Encoded using **OneHotEncoder** to safely convert them into a numerical format suitable for Machine Learning models without assuming ordinal relationships.
- Predictor variables will be reviewed for collinearity to avoid redundancy in the model.

## 2. Feature Selection & Engineering
- **Engineering Longitudinal Features:** The data contains time-series elements. Crucial derived features like `weight_change`, `whz_change`, and `illness_count_roll` (cumulative illnesses over the last 5 visits) will be engineered to analyze growth trajectories.
- **Predictor Variables:** The model will utilize 21 comprehensive features covering multiple dimensions:
   - **Growth & Trends:** `weight`, `height`, `WHZ` (Weight-for-Height Z-score), `weight_change`, `whz_change`
   - **Demographics:** `gender`, `birth_weight`
   - **Health & Clinical History:** `illness_count_roll`, `illness_count_last_month`, `immunization_status` (zero dose, partially immunized, etc.), `hiv_exposure`, `chronic_illness`, `congenital_disease`, `recurrent_diarrhea`
   - **Feeding Practices:** `exclusive_breastfeeding_6m`, `feeding_diversity_score`, `meal_frequency_per_day`
   - **Socio-Economic & WASH Metrics:** `household_income_level`, `parent_education_level`, `access_to_clean_water`, `sanitation_access`
- Variables are grouped logically into a comprehensive preprocessing pipeline (`ColumnTransformer`) to avoid overfitting and reduce model complexity.
- **Feature Importance:** Assessed using the built-in Random Forest feature importance functionality to identify the most influential predictors of malnutrition risk.

## 3. Model Training
- The **Random Forest Classifier** (using 100 decision trees) will be utilized as the core algorithm for predicting malnutrition risk based on the hybrid clinical setup.
- The dataset will be randomly split into **training (80%) and testing (20%) sets** to ensure robust out-of-sample validation.
- A unified ML tracking **Pipeline** will couple the data preprocessing (imputing and scaling) directly with the Random Forest classifier to prevent data leakage during training.

## 4. Model Evaluation
The performance of the Random Forest model will be evaluated heavily using the 20% testing dataset partition. Evaluation metrics will include:
- **Accuracy**: The overall proportion of correct malnutrition risk predictions.
- **Sensitivity (Recall)**: The proportion of actual malnourished children correctly identified (critical for clinical safety).
- **Specificity**: The proportion of healthy children correctly identified to minimize false alarms.
- **Precision**: The proportion of predicted malnourished cases that truly are malnourished.
- **Area Under the Receiver Operating Characteristic Curve (ROC-AUC)**: The overall discrimination ability of the model across different threshold levels.
- Visual evaluation standard output will include the **Confusion Matrix** and the **ROC Curve**.

## 5. Software and Reporting
- All core analysis will be conducted in Python using interactive data science workflows (e.g., Jupyter Notebooks and standard Python scripts), which allows active execution of code, interactive visualization, and automated workflow reporting.
- The user-facing clinical assessment tool will be deployed using the **Streamlit** framework to provide an interactive dashboard for healthcare workers.
- Outputs such as confusion matrices, ROC curves, feature importance plots, and performance metrics will be exported in high-quality formats for inclusion in the final research reports.
- The entire workflow is documented within the project workspace, ensuring reproducibility, transparency, and clarity for review and publication.

## 6. Data Security and Confidentiality
- All datasets (both real medical records and synthetic generations) will be **de-identified** before any analysis.
- Access to the raw data and the analysis platform will be strictly restricted to study team members.
- Results and findings will be reported strictly in aggregate form to protect participant privacy.
