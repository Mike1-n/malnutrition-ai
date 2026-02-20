import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# Page Configuration - Set to wide mode for maximum visibility
st.set_page_config(
    page_title="Child Malnutrition Longitudinal Analysis",
    page_icon="🏥",
    layout="wide"
)

# --- 1. Load Model & Standards ---
# @st.cache_resource
def load_model():
    """Loads the trained Random Forest model."""
    model_path = 'malnutrition_model.pkl'
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

@st.cache_data
def load_who_standards():
    """Loads WHO standards from CSV."""
    file_path = 'who_standards.csv'
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)

model = load_model()

def calculate_whz(height, weight, gender):
    """Calculates WHZ using WHO LMS method."""
    standards = load_who_standards()
    if standards is None:
        return None, "Data Missing"
    
    df = standards[standards['gender'] == gender]
    nearest_row = df.iloc[(df['height'] - height).abs().argsort()[:1]]
    
    if nearest_row.empty:
        return None, "Height Out of Range"
        
    L, M, S = nearest_row['L'].values[0], nearest_row['M'].values[0], nearest_row['S'].values[0]
    
    try:
        z_score = ((weight / M)**L - 1) / (L * S)
    except:
        return None, "Calc Error"
        
    # Classification Rules from JSON Spec
    # severe_wasting: whz <= -3
    # moderate_wasting: -3 < whz <= -2
    # normal: whz > -2
    
    if z_score <= -3: 
        category = "Severe Wasting"
    elif -3 < z_score <= -2: 
        category = "Moderate Wasting"
    else: 
        category = "Normal"

    return round(z_score, 2), category

# --- 2. Styles (High Visibility Theme) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    :root {
        --primary-color: #5dade2; /* Lighter Blue for dark mode */
        --secondary-color: #48c9b0; /* Lighter Teal */
        --background-color: #0e1117; /* Very Dark Grey/Black */
        --text-color: #ffffff; /* White Text */
        --card-bg: #262730; /* Dark Card Background */
        --danger-color: #ff6b6b; /* Bright Red */
        --warning-color: #ffa502; /* Bright Orange */
        --success-color: #2ed573; /* Bright Green */
    }

    /* Global */
    .stApp {
        background-color: var(--background-color);
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color) !important;
        font-weight: 700;
    }
    
    p, label, .stMarkdown, .stMetricLabel, [data-testid="stMarkdownContainer"] p {
        color: var(--text-color) !important;
    }
    
    /* Fix for Data Editor / DataFrame Visibility */
    [data-testid="stDataFrame"] {
        color: var(--text-color) !important;
    }
    [data-testid="stDataFrame"] svg {
        fill: var(--text-color) !important;
    }
    [data-testid="stDataFrame"] button {
        color: var(--text-color) !important;
    }

    /* Cards */
    .input-card {
        background-color: var(--card-bg);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-top: 5px solid var(--primary-color);
        margin-bottom: 20px;
    }

    /* Metrics */
    .metric-container {
        background-color: var(--card-bg);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #eee;
        transition: transform 0.2s;
    }
    .metric-container:hover {
        transform: translateY(-5px);
    }
    .label-text {
        font-size: 0.9rem;
        font-weight: 600;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
        margin-top: 10px;
    }
    .status-normal { color: #fff; background-color: var(--success-color); }
    .status-warning { color: #fff; background-color: var(--warning-color); }
    .status-danger { color: #fff; background-color: var(--danger-color); }
    .status-neutral { color: #fff; background-color: #95a5a6; }

    /* Custom Button */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary-color) 0%, #2980b9 100%);
        color: white !important;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
    }
    .stButton>button p { color: white !important; }

    /* Recommendation Box */
    .rec-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        color: var(--text-color); /* Ensure text is visible */
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); /* Darker shadow */
        background-color: var(--card-bg); /* Use dark card bg */
    }
    .rec-critical {
        border-left: 6px solid var(--danger-color);
        background: linear-gradient(90deg, rgba(255, 107, 107, 0.1) 0%, rgba(255, 107, 107, 0.0) 100%);
    }
    .rec-warning {
        border-left: 6px solid var(--warning-color);
        background: linear-gradient(90deg, rgba(255, 165, 2, 0.1) 0%, rgba(255, 165, 2, 0.0) 100%);
    }
    .rec-success {
        border-left: 6px solid var(--success-color);
        background: linear-gradient(90deg, rgba(46, 213, 115, 0.1) 0%, rgba(46, 213, 115, 0.0) 100%);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: var(--card-bg); /* Dark background */
        border-radius: 8px;
        font-weight: 600;
        color: var(--primary-color);
    }
    
    /* Progress Bar */
    .progress-bar-container {
        width: 100%;
        background-color: #333; /* Dark track */
        border-radius: 10px;
        height: 8px;
        margin-top: 8px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. Header ---
st.markdown("""
<div style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); padding: 30px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px; box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);">
    <h1 style="color: white !important; margin-bottom: 10px;">🏥 Child Malnutrition Longitudinal Analysis</h1>
    <p style="color: rgba(255,255,255,0.9) !important; font-size: 1.1rem;">AI-Powered Trend Risk Prediction & Clinical Decision Support</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model not found. Please train the model first.")
else:
    # --- 4. Main Layout ---
    left_col, right_col = st.columns([1.2, 1], gap="large")

    with left_col:
        st.subheader("📝 Patient Assessment")
        
        # Sections for better organization (Single Page View)
        
        with st.expander("1. 📏 Growth Metrics (Current & History)", expanded=True):
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1: gender = st.selectbox("Gender", ["Male", "Female"])
            with c2: current_age = st.number_input("Age (months)", 0, 60, 24)
            
            c3, c4 = st.columns(2)
            # Validation Rule: Weight [2, 30]
            with c3: birth_weight = st.number_input("Birth Weight (kg)", 0.5, 6.0, 3.0, step=0.1)
            with c4: st.write("") # Spacer

            st.markdown("<b>Longitudinal Data (Last 5 Visits):</b>", unsafe_allow_html=True)
            # Default Data for Editor
            default_data = pd.DataFrame({
                'Visit Date': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-02-01'), pd.Timestamp('2023-03-01'), pd.Timestamp('2023-04-01'), pd.Timestamp('2023-05-01')],
                'Weight (kg)': [9.0, 9.2, 9.1, 9.3, 9.5],
                'Height (cm)': [75.0, 75.5, 76.0, 76.5, 77.0],
                'Illness (Yes/No)': [False, False, True, False, False]
            })
            
            # Validation Rules: Height [45, 120], Weight [2, 30]
            edited_df = st.data_editor(
                default_data, 
                num_rows="dynamic", 
                column_config={
                    "Visit Date": st.column_config.DateColumn("Visit Date", format="YYYY-MM-DD"),
                    "Weight (kg)": st.column_config.NumberColumn("Weight", min_value=2, max_value=30, step=0.1),
                    "Height (cm)": st.column_config.NumberColumn("Height", min_value=45, max_value=120, step=0.1),
                    "Illness (Yes/No)": st.column_config.CheckboxColumn("Illness?", help="Was child ill at visit?")
                },
                width="stretch"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("2. 💊 Health & Clinical History", expanded=True):
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                # Dynamic Immunization Options based on Age
                imm_options = ["Age Appropriate", "Partially Immunized", "Zero Dose"]
                if current_age >= 12:
                    imm_options.insert(1, "Fully Immunized (12+ months)")
                
                imm_label = st.selectbox("Immunization Status", imm_options, index=0)
                
                # Map back to model value
                imm_map = {
                    "Age Appropriate": "age_appropriate",
                    "Fully Immunized (12+ months)": "fully_immunized",
                    "Partially Immunized": "partially_immunized",
                    "Zero Dose": "zero_dose"
                }
                immunization_status = imm_map[imm_label]

                recurrent_diarrhea = st.selectbox("Recurrent Diarrhea?", ["yes", "no"], index=1)
                chronic_illness = st.selectbox("Chronic Illness?", ["yes", "no"], index=1)
            with col_b:
                # HIV Status
                hiv_options = ["HIV Unexposed", "HIV Exposed Unaffected", "HIV Infected", "Unknown"]
                hiv_label = st.selectbox("HIV Status", hiv_options, index=0)
                
                # Map back to model value
                hiv_map = {
                    "HIV Unexposed": "hiv_unexposed",
                    "HIV Exposed Unaffected": "hiv_exposed_unaffected",
                    "HIV Infected": "hiv_infected",
                    "Unknown": "unknown"
                }
                hiv_exposure = hiv_map[hiv_label]
                
                congenital_disease = st.selectbox("Congenital Disease?", ["yes", "no"], index=1)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("3. 🍲 Feeding Practices", expanded=True):
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            col_c, col_d = st.columns(2)
            with col_c:
                # Exclusive Breastfeeding Duration Input
                ebf_duration = st.number_input("Duration of Exclusive Breastfeeding (months)", min_value=0, max_value=12, value=6, help="How many months was the child exclusively breastfed?")
                
                # Map to model input (yes if >= 6 months, else no)
                breastfeeding_6m = "yes" if ebf_duration >= 6 else "no"
                
                # meal_freq input removed from UI as per user request
                meal_freq = 3 # Default value for model compatibility
            with col_d:
                if current_age >= 6:
                    st.markdown('<p style="color:var(--text-color); font-weight:600;">Complementary Feeding (Select all consumed yesterday):</p>', unsafe_allow_html=True)
                    
                    nutrient_options = [
                        "Grains, Roots, Tubers",
                        "Legumes & Nuts",
                        "Dairy Products",
                        "Flesh Foods",
                        "Eggs",
                        "Vitamin A rich fruits/vegetables",
                        "Other Fruits"
                    ]
                    
                    selected_nutrients = st.multiselect("Select Nutrients", nutrient_options, label_visibility="collapsed")
                    feeding_diversity = len(selected_nutrients)
                    st.caption(f"Calculated Diversity Score: **{feeding_diversity}/7**")
                else:
                    st.info(f"ℹ️ Complementary feeding is recommended starting at 6 months. (Current Age: {current_age}m)")
                    feeding_diversity = 0 # Not applicable yet
                    selected_nutrients = []

            st.markdown('</div>', unsafe_allow_html=True)
            
        with st.expander("4. 🏠 Socio-Economic Factors (SES Score)", expanded=True):
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            col_e, col_f = st.columns(2)
            
            with col_e:
                # 1. Education (0, 1, 2, 4)
                edu_options = ["No formal education", "Primary education", "Secondary education", "College / University"]
                edu_input = st.selectbox("Caregiver Education", edu_options, index=2)
                
                edu_points_map = {
                    "No formal education": 0,
                    "Primary education": 1,
                    "Secondary education": 2,
                    "College / University": 4
                }
                edu_score = edu_points_map[edu_input]
                
                # Map for model input
                edu_model_map = {
                    "No formal education": "none",
                    "Primary education": "primary",
                    "Secondary education": "secondary",
                    "College / University": "tertiary"
                }
                education_level = edu_model_map[edu_input]

                # 2. Occupation (0, 1, 2, 3)
                occ_options = ["Unemployed", "Casual labourer", "Small business", "Formal employment / Professional"]
                occ_input = st.selectbox("Caregiver Occupation", occ_options, index=2)
                occ_points_map = {
                    "Unemployed": 0,
                    "Casual labourer": 1,
                    "Small business": 2,
                    "Formal employment / Professional": 3
                }
                occ_score = occ_points_map[occ_input]

            with col_f:
                # 3. Household Assets (1 point each)
                # "household assets (electricity(1),piped water(1),refrigiretor (1),television(1))"
                assets_options = ["Electricity", "Piped Water", "Refrigerator", "Television"]
                selected_assets = st.multiselect("Household Assets (Select all that apply)", assets_options)
                assets_score = len(selected_assets) * 1
                
                # Infer water access (for model/recommendations) if Piped Water is selected, 
                # but allow manual override if they have other clean water access
                has_piped_water = "Piped Water" in selected_assets
                
                # 4. Household Crowding (0, 1, 2)
                # ">3persons per room(0),2-3 persons per room(1),<2 persons per room(2)"
                crowding_options = ["> 3 persons per room", "2 - 3 persons per room", "< 2 persons per room"]
                crowding_input = st.selectbox("Household Crowding", crowding_options, index=1)
                crowding_points_map = {
                    "> 3 persons per room": 0,
                    "2 - 3 persons per room": 1,
                    "< 2 persons per room": 2
                }
                crowding_score = crowding_points_map[crowding_input]

            # Calculate SES Score
            ses_score_total = edu_score + occ_score + assets_score + crowding_score
            
            # Determine SES Category and Model mapping
            # <5 (low ses), 5-8 (middle ses) >9 (actually >=9) (high ses)
            # Max score: 4 + 3 + 4 + 2 = 13. User said max 12, but math allows 13. 
            # Logic: < 5, 5-8, > 8 (9+)
            
            if ses_score_total < 5:
                ses_category_label = "Low SES"
                income_level = "low" # Model mapping
                ses_color = "#e74c3c" # Red
            elif 5 <= ses_score_total <= 8:
                ses_category_label = "Middle SES"
                income_level = "middle" # Model mapping
                ses_color = "#f39c12" # Orange
            else:
                ses_category_label = "High SES"
                income_level = "high" # Model mapping
                ses_color = "#2ecc71" # Green

            st.markdown("---")
            c_score, c_cat = st.columns([1, 2])
            with c_score:
                st.markdown(f"**Total SES Score:** <span style='font-size:1.2em'>{ses_score_total} / 13</span>", unsafe_allow_html=True)
            with c_cat:
                st.markdown(f"**SES Category:** <span style='color:{ses_color}; font-weight:bold; font-size:1.2em'>{ses_category_label}</span>", unsafe_allow_html=True)
            
            # Additional water/sanitation inputs for model completeness if not covered by assets
            with st.expander("Additional WASH Factors (Optional)", expanded=False):
                water_options = ["yes", "no"]
                water_default = 0 if has_piped_water else 1
                water_access = st.selectbox("Access to Clean Water?", water_options, index=water_default)
                sanitation_access = st.selectbox("Access to Sanitation?", ["yes", "no"], index=0)
            
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 ANALYZE COMPREHENSIVE RISK", type="primary")

    with right_col:
        st.subheader("📊 Clinical Assessment Results")
        
        if predict_btn and not edited_df.empty:
            # Process Data
            df = pd.DataFrame(edited_df)
            df['Visit Date'] = pd.to_datetime(df['Visit Date'], errors='coerce')
            df = df.sort_values(by='Visit Date')
            
            # --- Feature Engineering for Model ---
            # Calculate WHZ for all rows first
            whz_scores = []
            for index, row in df.iterrows():
                z, cat = calculate_whz(row['Height (cm)'], row['Weight (kg)'], gender)
                whz_scores.append(z if z is not None else 0)
            df['WHZ'] = whz_scores

            # Current values (Latest row)
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest # Fallback if only 1 row
            
            current_weight = latest['Weight (kg)']
            current_height = latest['Height (cm)']
            current_whz = latest['WHZ']
            
            # Derived Features (Lag/Roll)
            weight_change = current_weight - prev['Weight (kg)'] if len(df) > 1 else 0
            whz_change = current_whz - prev['WHZ'] if len(df) > 1 else 0
            illness_count_roll = df['Illness (Yes/No)'].sum() # Simple sum of checked boxes

            # Input Vector for Model
            # Needs to match the training data columns (minus the dropped ones)
            # The pipeline handles encoding, so we pass raw values.
            
            input_vector = pd.DataFrame([{
                'weight': current_weight,
                'height': current_height,
                'WHZ': current_whz,
                'gender': gender,
                'birth_weight': birth_weight,
                'weight_change': weight_change,
                'whz_change': whz_change,
                'illness_count_roll': illness_count_roll,
                'illness_count_last_month': illness_count_roll, # Proxying from longitudinal data for now
                'immunization_status': immunization_status,
                'hiv_exposure': hiv_exposure,
                'chronic_illness': chronic_illness,
                'congenital_disease': congenital_disease,
                'recurrent_diarrhea': recurrent_diarrhea,
                'exclusive_breastfeeding_6m': breastfeeding_6m,
                'feeding_diversity_score': feeding_diversity,
                'meal_frequency_per_day': meal_freq,
                'household_income_level': income_level,
                'parent_education_level': education_level,
                'access_to_clean_water': water_access,
                'sanitation_access': sanitation_access
            }])
            
            # Prediction
            try:
                prob = model.predict_proba(input_vector)[0][1]
                if prob < 0.3: ml_risk, ml_class = "Low Risk", "status-normal"
                elif prob < 0.7: ml_risk, ml_class = "Moderate Risk", "status-warning"
                else: ml_risk, ml_class = "High Risk", "status-danger"
            except Exception as e:
                prob, ml_risk, ml_class = 0, "Error", "status-neutral"
                st.error(f"Prediction Error: {e}")

            # Clinical Calcs (Current)
            # bmi_proxy removed as per user request
            z_score, whz_category = calculate_whz(current_height, current_weight, gender)
            
            # Display Grid
            r1_col1, r1_col2 = st.columns(2)
            with r1_col1:
                # Progress Bar Color Logic
                if prob < 0.3: bar_color = "#2ecc71" # Green
                elif prob < 0.7: bar_color = "#f39c12" # Orange
                else: bar_color = "#e74c3c" # Red
                
                st.markdown(f"""
                <div class="metric-container">
                    <p class="label-text">🤖 Trend Prediction</p>
                    <div class="status-badge {ml_class}">{ml_risk}</div>
                    <p style="margin-top: 15px; color: #b0b0b0; font-size: 0.9rem;">Probability: <b style="color: #ffffff; font-size: 1.2rem; font-weight: 800;">{prob:.1%}</b></p>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" style="width: {prob*100}%; background-color: {bar_color};"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
            with r1_col2:
                # Z-score display logic
                # whz_category already contains "Severe Wasting" or "Moderate Wasting" or "Normal"
                
                whz_risk_class = "status-normal"
                if whz_category == "Severe Wasting":
                    whz_risk_class = "status-danger"
                elif whz_category == "Moderate Wasting":
                    whz_risk_class = "status-warning"
                
                whz_val_str = f"{z_score:.2f}" if z_score is not None else "N/A"
                
                st.markdown(f"""
                <div class="metric-container">
                    <p class="label-text">📏 Weight-for-Height Z-Score</p>
                    <div class="status-badge {whz_risk_class}">{whz_category}</div>
                    <p style="margin-top: 15px; color: #b0b0b0; font-size: 0.9rem;">Value: <b style="color: #ffffff; font-size: 1.2rem; font-weight: 800;">{whz_val_str} SD</b></p>
                </div>""", unsafe_allow_html=True)
            
            # Weight & BMI & WHZ Trend Charts
            st.markdown("### 📈 Growth Trends")
            
            # Calculate BMI and WHZ for all visits
            df['Height_m'] = df['Height (cm)'] / 100
            # Drop rows with invalid dates
            df = df.dropna(subset=['Visit Date'])

            # Calculate WHZ for all rows first (already done above)
            
            # --- Remove BMI Calculation ---
            # df['BMI'] = ... (Removed)
            
            # Function to plot with non-overlapping labels
            # Function to plot with non-overlapping labels
            def plot_trend(data, x_col, y_col, title, color, y_label):
                if data.empty: return None
                
                # Dark Mode Styles
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('none') # Transparent bg
                ax.set_facecolor('none')      # Transparent plot area
                
                # Plot Line and Points
                ax.plot(data[x_col], data[y_col], marker='o', linestyle='-', color=color, linewidth=2, markersize=8)
                
                # Add Data Labels with alternating offset to prevent overlap
                for i, txt in enumerate(data[y_col]):
                    if pd.isna(data[y_col].iloc[i]): continue
                    offset = 15 if i % 2 == 0 else -25  # Alternate up/down
                    val_str = f"{txt:.1f}"
                    ax.annotate(val_str, (data[x_col].iloc[i], data[y_col].iloc[i]),
                                textcoords="offset points", xytext=(0, offset), ha='center',
                                fontsize=9, fontweight='bold', color='white',
                                arrowprops=dict(arrowstyle="-", color='white', alpha=0.5))

                # Formatting
                ax.set_title(title, fontweight='bold', color='white')
                ax.set_ylabel(y_label, fontweight='bold', color='white')
                ax.set_xlabel('Visit Date', fontweight='bold', color='white')
                ax.grid(True, linestyle='--', alpha=0.3, color='white')
                
                # Axis Colors
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                for spine in ax.spines.values():
                    spine.set_color('white')
                
                # Rotate Date Labels
                fig.autofmt_xdate(rotation=45)
                
                # Ensure y-axis has some padding for labels
                if not data[y_col].dropna().empty:
                    y_min, y_max = data[y_col].min(), data[y_col].max()
                    margin = (y_max - y_min) * 0.2 if y_max != y_min else 1.0
                    ax.set_ylim(y_min - margin, y_max + margin)
                
                return fig

            # Layout Charts
            # Row 1: Weight (Full Width)
            st.markdown("####") 
            # Use Brighter Colors for Dark Mode
            fig_weight = plot_trend(df, 'Visit Date', 'Weight (kg)', 'Weight Trajectory', '#00d2d3', 'Weight (kg)') # Bright Cyan
            st.pyplot(fig_weight)

            # Row 2: WHZ (Full Width)
            st.markdown("####") # Spacer

            # Row 2: WHZ (Full Width or Centered)
            st.markdown("####") # Spacer
            fig_whz = plot_trend(df, 'Visit Date', 'WHZ', 'WHZ Score Trajectory', '#ff9f43', 'Z-Score') # Bright Orange
            st.pyplot(fig_whz)
            
            # Recommendations
            st.markdown("### 🩺 Clinical Recommendations")
            
            risk_factors = []
            
            # 1. Check WHZ (Primary Indicator now)
            # 1. Check WHZ (Primary Indicator now)
            if z_score is not None:
                if z_score <= -3:
                    risk_factors.append("Severe Wasting (WHZ <= -3 SD)")
                elif -3 < z_score <= -2:
                    risk_factors.append("Moderate Wasting (-3 < WHZ <= -2 SD)")
            
            # 2. Check Weight Loss Trend
            
            # 3. Check Weight Loss Trend
            if weight_change < 0:
                risk_factors.append(f"Recent Weight Loss (-{abs(weight_change):.2f}kg)")
                
            # 4. Check ML Prediction
            # 4. Check ML Prediction
            if ml_risk == "High Risk":
                risk_factors.append("ML Model predicts High Risk (Multifactorial)")
            
            # 5. Check New Risk Factors
            if immunization_status == 'zero_dose':
                 risk_factors.append("❌ Zero Dose: Immediate Vaccination Referral Required")
            elif immunization_status == 'partially_immunized':
                 risk_factors.append("⚠️ Partially Immunized: Refer for Catch-up Counseling")
            
            # HIV Risk
            if hiv_exposure == 'hiv_infected':
                risk_factors.append("❌ HIV Infected: High Risk - Immediate Clinical Management Required")
            elif hiv_exposure == 'hiv_exposed_unaffected':
                risk_factors.append("⚠️ HIV Exposed Unaffected: Moderate Risk - Monitor Growth Closely")
            elif hiv_exposure == 'unknown':
                 risk_factors.append("⚠️ HIV Status Unknown: Recommend Testing if indicated")

            # Breastfeeding Risk Logic
            if ebf_duration < 2:
                risk_factors.append(f"❌ High Risk: Exclusive Breastfeeding stopped too early (< 2 months)")
            elif 2 <= ebf_duration <= 5:
                risk_factors.append(f"⚠️ Moderate Risk: Exclusive Breastfeeding stopped early (2-5 months)")
            elif ebf_duration == 6:
                pass # Low Risk / Standard
            elif ebf_duration >= 6:
                pass # User Note: Exclusive breastfeeding was done (Optimal)

            # Only check diversity if child is of complementary feeding age (>= 6 months)
            if current_age >= 6 and feeding_diversity < 4:
                risk_factors.append("Low Dietary Diversity (< 4 groups)")
            
            if water_access == 'no':
                risk_factors.append("No Access to Clean Water (Risk of Infection)")

            # SES Risk
            if ses_category_label == "Low SES":
                risk_factors.append(f"Low Socio-Economic Status (Score: {ses_score_total}/12): High Malnutrition Risk")
            elif ses_category_label == "Middle SES":
                 risk_factors.append(f"Middle Socio-Economic Status (Score: {ses_score_total}/12): Moderate Risk")

            # Construct Message
            if not risk_factors:
                rec_text = "✅ Child is growing well. Maintain healthy feeding practices."
                rec_class = "rec-success"
            else:
                rec_text = "⚠️ **CRITICAL FINDINGS:**<br>" + "<br>".join([f"- {factor}" for factor in risk_factors])
                rec_text += "<br><br><b>ACTION: Immediate clinical assessment and referral required.</b>"
                rec_class = "rec-critical" if ((z_score is not None and z_score < -3) or ml_risk == "High Risk") else "rec-warning"

            st.markdown(f"""<div class="rec-box {rec_class}"><p style="font-size: 1.1rem; margin-bottom: 0;">{rec_text}</p></div>""", unsafe_allow_html=True)
            
        else:
            st.info("👈 Please enter the last 5 session records in the table to analyze.")
