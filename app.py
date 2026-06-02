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
        
    # WHO Classification Rules (0–5 years)
    # < -3 SD : Severely Wasted
    # < -2 SD : Wasted (Moderately)
    # >-2 and <+1 SD : Normal
    # > +1 SD : At Risk of Overweight
    # > +2 SD : Overweight
    # > +3 SD : Obese

    if z_score < -3:
        category = "Severely Wasted"
    elif z_score < -2:
        category = "Wasted"
    elif z_score <= 1:
        category = "Normal"
    elif z_score <= 2:
        category = "At Risk of Overweight"
    elif z_score <= 3:
        category = "Overweight"
    else:
        category = "Obese"

    return round(z_score, 2), category

# --- 2. Styles & Theme ---
if 'theme_toggle' not in st.session_state:
    st.session_state.theme_toggle = True

st.session_state.theme = 'dark' if st.session_state.theme_toggle else 'light'

if st.session_state.theme == 'dark':
    theme_css = """
    :root {
        --primary-color: #5dade2; 
        --secondary-color: #48c9b0; 
        --background-color: #0e1117; 
        --text-color: #ffffff; 
        --text-muted: #b0b0b0;
        --card-bg: #262730; 
        --danger-color: #ff6b6b; 
        --warning-color: #ffa502; 
        --success-color: #2ed573;
        --metric-border: #333333;
    }
    """
else:
    theme_css = """
    :root {
        --primary-color: #2980b9; 
        --secondary-color: #16a085; 
        --background-color: #f0f2f6; 
        --text-color: #1e1e1e; 
        --text-muted: #666666;
        --card-bg: #ffffff; 
        --danger-color: #e74c3c; 
        --warning-color: #f39c12; 
        --success-color: #27ae60;
        --metric-border: #e0e0e0;
    }
    """

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    {theme_css}


    /* Global */
    .stApp {{
        background-color: var(--background-color);
        font-family: 'Inter', sans-serif;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-color) !important;
        font-weight: 700;
    }}
    
    p, label, .stMarkdown, .stMetricLabel, [data-testid="stMarkdownContainer"] p {{
        color: var(--text-color) !important;
    }}
    
    /* Fix for Data Editor / DataFrame Visibility */
    [data-testid="stDataFrame"] {{
        color: var(--text-color) !important;
    }}
    [data-testid="stDataFrame"] svg {{
        fill: var(--text-color) !important;
    }}
    [data-testid="stDataFrame"] button {{
        color: var(--text-color) !important;
    }}



    /* Metrics */
    .metric-container {{
        background-color: var(--card-bg);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid var(--metric-border);
        transition: transform 0.2s;
    }}
    .metric-container:hover {{
        transform: translateY(-5px);
    }}
    .label-text {{
        font-size: 0.9rem;
        font-weight: 600;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    /* Status Badges */
    .status-badge {{
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
        margin-top: 10px;
    }}
    .status-normal {{ color: #fff; background-color: var(--success-color); }}
    .status-warning {{ color: #fff; background-color: var(--warning-color); }}
    .status-danger {{ color: #fff; background-color: var(--danger-color); }}
    .status-neutral {{ color: #fff; background-color: #95a5a6; }}

    /* Custom Button */
    .stButton>button {{
        background: linear-gradient(135deg, var(--primary-color) 0%, #2980b9 100%);
        color: white !important;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
    }}
    .stButton>button p {{ color: white !important; }}

    /* Recommendation Box */
    .rec-box {{
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        color: var(--text-color); /* Ensure text is visible */
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); /* Darker shadow */
        background-color: var(--card-bg); /* Use dark card bg */
    }}
    .rec-critical {{
        border-left: 6px solid var(--danger-color);
        background: linear-gradient(90deg, rgba(255, 107, 107, 0.1) 0%, rgba(255, 107, 107, 0.0) 100%);
    }}
    .rec-warning {{
        border-left: 6px solid var(--warning-color);
        background: linear-gradient(90deg, rgba(255, 165, 2, 0.1) 0%, rgba(255, 165, 2, 0.0) 100%);
    }}
    .rec-success {{
        border-left: 6px solid var(--success-color);
        background: linear-gradient(90deg, rgba(46, 213, 115, 0.1) 0%, rgba(46, 213, 115, 0.0) 100%);
    }}
    
    /* Expander Styling */
    .streamlit-expanderHeader {{
        background-color: var(--card-bg); /* Dark background */
        border-radius: 8px;
        font-weight: 600;
        color: var(--primary-color);
    }}
    
    /* Progress Bar */
    .progress-bar-container {{
        width: 100%;
        background-color: #333; /* Dark track */
        border-radius: 10px;
        height: 8px;
        margin-top: 8px;
        overflow: hidden;
    }}
    .progress-bar-fill {{
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }}
    </style>
""", unsafe_allow_html=True)

# --- 3. Header ---
head_col, tog_col = st.columns([6, 1])

with tog_col:
    st.write("") # Margin top
    st.toggle("🌙 Mode", key="theme_toggle")

with head_col:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); padding: 25px; border-radius: 15px; color: white; text-align: center; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);">
        <h1 style="color: white !important; margin-bottom: 5px; font-size: 2.2rem;">🏥 Child Malnutrition Analysis</h1>
        <p style="color: rgba(255,255,255,0.9) !important; font-size: 1.1rem; margin-bottom: 0;">AI-Powered Trend Risk Prediction & Clinical Decision Support</p>
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
        
        with st.expander("1. 📏 Growth Metrics (Current)", expanded=True):
            c1, c2 = st.columns(2)
            with c1: gender = st.selectbox("Gender", ["Male", "Female"])
            with c2: current_age = st.number_input("Age (months)", 0, 60, 24)
            
            c3, c4 = st.columns(2)
            # Validation Rule: Weight [2, 30]
            with c3: birth_weight = st.number_input("Birth Weight (kg)", 0.5, 6.0, 3.0, step=0.1)
            with c4: illness_count_last_month = st.number_input("Illnesses in Last Month", 0, 10, 0)

            st.markdown("<b>Current Measurements:</b>", unsafe_allow_html=True)
            c5, c6, c7 = st.columns(3)
            with c5: current_weight = st.number_input("Current Weight (kg)", min_value=2.0, max_value=30.0, value=9.5, step=0.1)
            with c6: current_height = st.number_input("Current Height (cm)", min_value=45.0, max_value=120.0, value=75.0, step=0.1)
            with c7: muac_mm = st.number_input("MUAC (mm)", min_value=80, max_value=200, value=135, step=1)

        with st.expander("2. 💊 Health & Clinical History", expanded=True):
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

        with st.expander("3. 🍲 Feeding Practices", expanded=True):
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

            
        with st.expander("4. 🏠 Socio-Economic Factors (SES Score)", expanded=True):
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
            
            # Auto-infer WASH factors from assets
            water_access = "yes" if has_piped_water else "no"
            sanitation_access = "yes" if ses_score_total >= 5 else "no"  # Reasonable proxy given data
            

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 ANALYZE COMPREHENSIVE RISK", type="primary")

    with right_col:
        st.subheader("📊 Clinical Assessment Results")
        
        if predict_btn:
            # Clinical Calcs (Current)
            current_whz, whz_category = calculate_whz(current_height, current_weight, gender)
            z_score = current_whz

            # Input Vector for Model
            input_vector = pd.DataFrame([{
                'age_months': current_age,
                'weight': current_weight,
                'height': current_height,
                'muac_mm': muac_mm,
                'gender': gender,
                'birth_weight': birth_weight,
                'household_income_level': income_level,
                'parent_education_level': education_level,
                'access_to_clean_water': water_access,
                'sanitation_access': sanitation_access,
                'hiv_exposure': hiv_exposure,
                'chronic_illness': chronic_illness,
                'congenital_disease': congenital_disease,
                'recurrent_diarrhea': recurrent_diarrhea,
                'exclusive_breastfeeding_6m': breastfeeding_6m,
                'immunization_status': immunization_status,
                'feeding_practice': "Mixed Feeding" if feeding_diversity > 3 else "Complementary Feeding" if current_age >= 6 else "Exclusive Breastfeeding",
                'recent_illness': "no" if illness_count_last_month == 0 else "Fever",
            }])

            # Prediction (Multi-class)
            try:
                # The model now outputs 5 classes instead of a binary probability
                predicted_class = model.predict(input_vector)[0]
                probabilities = model.predict_proba(input_vector)[0]
                classes = model.classes_
                prob = probabilities[list(classes).index(predicted_class)]
                
                ml_risk = predicted_class
                
                # Determine styling based on class
                if "Severe" in ml_risk:
                    ml_class = "status-danger"
                    bar_color = "#e74c3c" # Red
                elif "Moderate Malnutrition" in ml_risk or "High Risk" in ml_risk:
                    ml_class = "status-warning"
                    bar_color = "#f39c12" # Orange
                elif "Moderate Risk" in ml_risk:
                    ml_class = "status-warning"
                    bar_color = "#f1c40f" # Yellow
                else:
                    ml_class = "status-normal"
                    bar_color = "#2ecc71" # Green
                    
            except Exception as e:
                prob, ml_risk, ml_class, bar_color = 0, "Error", "status-neutral", "#95a5a6"
                st.error(f"Prediction Error: {e}")

            # Pre-calculate risk factors for explanation
            risk_factors = []
            contributing_factors = []
            
            if z_score is not None:
                if z_score < -3:
                    risk_factors.append("Severely Wasted (WHZ < -3 SD)")
                    contributing_factors.append("Critically low Weight-for-Height Z-Score")
                elif z_score < -2:
                    risk_factors.append("Wasted (Moderately) (-3 ≤ WHZ < -2 SD)")
                    contributing_factors.append("Low Weight-for-Height Z-Score")
                elif z_score > 1:
                    risk_factors.append("At Risk of Overweight (WHZ > +1 SD)")
                    contributing_factors.append("Elevated Weight-for-Height Z-Score")
                    if z_score > 2:
                        risk_factors[-1] = "Overweight (WHZ > +2 SD)"
                    if z_score > 3:
                        risk_factors[-1] = "Obese (WHZ > +3 SD)"
            
            if illness_count_last_month > 0:
                contributing_factors.append(f"Recent illness history ({illness_count_last_month} in last month)")
                if illness_count_last_month >= 3:
                     risk_factors.append("High frequency of recent illnesses")
                
            if immunization_status == 'zero_dose':
                 risk_factors.append("❌ Zero Dose: Immediate Vaccination Referral Required")
                 contributing_factors.append("Zero dose immunization status")
            elif immunization_status == 'partially_immunized':
                 risk_factors.append("⚠️ Partially Immunized: Refer for Catch-up Counseling")
                 contributing_factors.append("Partial immunization")
            
            if hiv_exposure == 'hiv_infected':
                risk_factors.append("❌ HIV Infected: High Risk - Immediate Clinical Management Required")
                contributing_factors.append("HIV Infected status")
            elif hiv_exposure == 'hiv_exposed_unaffected':
                risk_factors.append("⚠️ HIV Exposed Unaffected: Moderate Risk - Monitor Growth Closely")
                contributing_factors.append("HIV Exposed status")
            elif hiv_exposure == 'unknown':
                 risk_factors.append("⚠️ HIV Status Unknown: Recommend Testing if indicated")

            if ebf_duration < 2:
                risk_factors.append(f"❌ High Risk: Exclusive Breastfeeding stopped too early (< 2 months)")
                contributing_factors.append("Suboptimal exclusive breastfeeding duration")
            elif 2 <= ebf_duration <= 5:
                risk_factors.append(f"⚠️ Moderate Risk: Exclusive Breastfeeding stopped early (2-5 months)")
                contributing_factors.append("Early cessation of exclusive breastfeeding")
            
            if current_age >= 6 and feeding_diversity < 4:
                risk_factors.append("Low Dietary Diversity (< 4 groups)")
                contributing_factors.append("Low feeding diversity score")
            
            if water_access == 'no' or sanitation_access == 'no':
                risk_factors.append("Poor WASH conditions (Infection Risk)")
                contributing_factors.append("Limited access to clean water/sanitation")

            if ses_category_label == "Low SES":
                risk_factors.append(f"Low Socio-Economic Status (Score: {ses_score_total}/13): High Malnutrition Risk")
                contributing_factors.append("Lower socio-economic factors")

            if len(contributing_factors) == 0:
                contributing_factors.append("No major specific risk factors identified; percentage reflects baseline demographic/growth patterns.")

            if not risk_factors:
                rec_text = "✅ Child is growing well. Maintain healthy feeding practices."
                rec_class = "rec-success"
            else:
                rec_text = "⚠️ **CRITICAL FINDINGS:**<br>" + "<br>".join([f"- {factor}" for factor in risk_factors])
                if ((z_score is not None and z_score < -2) or ml_risk == "High Risk"):
                    rec_text += "<br><br><b>ACTION: Immediate clinical assessment and referral required.</b>"
                else:
                    rec_text += "<br><br><b>ACTION: Close monitoring and active nutritional support advised.</b>"
                rec_class = "rec-critical" if ((z_score is not None and z_score <= -3) or ml_risk == "High Risk") else "rec-warning"
            
            # Display Grid
            r1_col1, r1_col2 = st.columns(2)
            with r1_col1:
                contrib_html = "".join([f"<li style='color: var(--text-color); font-size: 0.85rem; margin-bottom: 3px;'>{cf}</li>" for cf in contributing_factors])

                st.markdown(f"""
                <div class="metric-container">
                    <p class="label-text">🤖 Trend Prediction</p>
                    <div class="status-badge {ml_class}">{ml_risk}</div>
                    <p style="margin-top: 15px; color: var(--text-muted); font-size: 0.9rem;">Confidence: <b style="color: var(--text-color); font-size: 1.2rem; font-weight: 800;">{prob:.1%}</b></p>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" style="width: {prob*100}%; background-color: {bar_color};"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
                
                st.markdown(f"""<div class="rec-box {rec_class}" style="margin-top: 10px; padding: 15px;"><p style="font-size: 1.0rem; margin-bottom: 0;"><b>Clinical Recommendation based on Risk Factors:</b><br><br>{rec_text}</p></div>""", unsafe_allow_html=True)
            with r1_col2:
                # Z-score display logic
                # Badge color based on WHO category
                if whz_category == "Severely Wasted":
                    whz_risk_class = "status-danger"
                elif whz_category in ("Wasted", "At Risk of Overweight", "Overweight"):
                    whz_risk_class = "status-warning"
                elif whz_category == "Obese":
                    whz_risk_class = "status-danger"
                else:
                    whz_risk_class = "status-normal"
                
                whz_val_str = f"{z_score:.2f}" if z_score is not None else "N/A"
                
                st.markdown(f"""
                <div class="metric-container">
                    <p class="label-text">📏 Weight-for-Height Z-Score</p>
                    <div class="status-badge {whz_risk_class}">{whz_category}</div>
                    <p style="margin-top: 15px; color: var(--text-muted); font-size: 0.9rem;">Value: <b style="color: var(--text-color); font-size: 1.2rem; font-weight: 800;">{whz_val_str} SD</b></p>
                </div>""", unsafe_allow_html=True)
            
        else:
            st.info("👈 Please enter the current measurements and click Analyze.")
