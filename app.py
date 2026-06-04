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

    if z_score <= -3:
        category = "Severe Acute Malnutrition"
    elif z_score <= -2:
        category = "Moderate Acute Malnutrition"
    elif z_score <= -1:
        category = "High Risk"
    elif z_score <= 1:
        category = "Moderate Risk"
    elif z_score < 3:
        category = "Not Malnutritioned - Low Risk"
    else:
        category = "Obese"

    return round(z_score, 2), category

def plot_whz_trajectory_with_regions(df):
    """Generates an interactive Altair chart showing the subject's WHZ trajectory with critical regions."""
    import altair as alt
    import pandas as pd

    df_sorted = df.sort_values(by="age_months").copy()
    
    # Domain calculations
    min_age = int(df_sorted["age_months"].min())
    max_age = int(df_sorted["age_months"].max())
    
    xmin = max(0, min_age - 3)
    xmax = max_age + 3
    
    regions = pd.DataFrame([
        {"ymin": -5.0, "ymax": -3.0, "color": "#e74c3c", "label": "SAM (≤ -3 SD)"},
        {"ymin": -3.0, "ymax": -2.0, "color": "#e67e22", "label": "MAM (-3 to -2 SD)"},
        {"ymin": -2.0, "ymax": -1.0, "color": "#f1c40f", "label": "High Risk (-2 to -1 SD)"},
        {"ymin": -1.0, "ymax": 1.0, "color": "#2ecc71", "label": "Normal (-1 to 1 SD)"},
        {"ymin": 1.0, "ymax": 3.0, "color": "#58d68d", "label": "Low Risk (1 to 3 SD)"},
        {"ymin": 3.0, "ymax": 5.0, "color": "#9b59b6", "label": "Obese (≥ 3 SD)"}
    ])
    
    # Theme configuration
    theme = st.session_state.get('theme', 'dark')
    text_color = '#ffffff' if theme == 'dark' else '#2c3e50'
    grid_color = '#444444' if theme == 'dark' else '#e0e0e0'
    
    rects = alt.Chart(regions).mark_rect(opacity=0.12).encode(
        y='ymin:Q',
        y2='ymax:Q',
        color=alt.Color('label:N', scale=alt.Scale(domain=list(regions["label"]), range=list(regions["color"])), legend=alt.Legend(title="WHO Regions", titleColor=text_color, labelColor=text_color))
    )
    
    boundaries = pd.DataFrame([{"y": -3.0}, {"y": -2.0}, {"y": -1.0}, {"y": 1.0}, {"y": 3.0}])
    rules = alt.Chart(boundaries).mark_rule(
        strokeDash=[3, 3],
        color='#95a5a6',
        size=1.0
    ).encode(
        y='y:Q'
    )
    
    line = alt.Chart(df_sorted).mark_line(
        color='#2980b9',
        size=3.0
    ).encode(
        x=alt.X('age_months:Q', title='Age (months)', scale=alt.Scale(domain=[xmin, xmax])),
        y=alt.Y('z_score:Q', title='WHZ (SD)', scale=alt.Scale(domain=[-4.5, 4.5]))
    )
    
    points = alt.Chart(df_sorted).mark_point(
        color='#2980b9',
        size=90,
        filled=True
    ).encode(
        x='age_months:Q',
        y='z_score:Q',
        tooltip=[
            alt.Tooltip('age_months:Q', title='Age (months)'),
            alt.Tooltip('z_score:Q', title='Z-Score (SD)', format='.2f'),
            alt.Tooltip('weight:Q', title='Weight (kg)'),
            alt.Tooltip('height:Q', title='Height (cm)'),
            alt.Tooltip('whz_category:N', title='Category')
        ]
    )
    
    # Text labels showing the Z-score value on top of each point
    labels = alt.Chart(df_sorted).mark_text(
        align='center',
        baseline='bottom',
        dy=-10,
        fontWeight='bold',
        color=text_color
    ).encode(
        x='age_months:Q',
        y='z_score:Q',
        text=alt.Text('z_score:Q', format='+.2f')
    )
    
    chart = alt.layer(rects, rules, line, points, labels).properties(
        height=320
    ).configure_view(
        strokeOpacity=0
    ).configure_axis(
        gridColor=grid_color,
        labelColor=text_color,
        titleColor=text_color,
        domainColor=text_color,
        tickColor=text_color
    ).configure_title(
        color=text_color,
        fontWeight='bold'
    ).interactive(
        bind_y=False # Lock Y-axis zoom/pan, zoom/pan *alone* on X-axis (Age) for best usability!
    )
    
    return chart

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

    /* Mobile Responsive Optimizations */
    @media (max-width: 768px) {{
        /* Force columns to stack vertically on phone screens */
        [data-testid="column"] {{
            width: 100% !important;
            flex: 1 1 100% !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
            margin-bottom: 12px !important;
        }}
        
        /* Scale down headings for mobile viewports */
        h1 {{
            font-size: 1.5rem !important;
        }}
        h2 {{
            font-size: 1.25rem !important;
        }}
        h3 {{
            font-size: 1.1rem !important;
        }}
        
        /* Reduce padding inside metric cards and recommendation boxes */
        .metric-container {{
            padding: 15px !important;
            margin-bottom: 12px !important;
        }}
        .status-badge {{
            font-size: 0.85rem !important;
            padding: 4px 12px !important;
        }}
        .rec-box {{
            padding: 12px !important;
            margin-top: 8px !important;
        }}
    }}
    </style>
""", unsafe_allow_html=True)

# --- 3. Sidebar Navigation & Connection ---
st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="margin: 0; font-size: 1.5rem; color: #3498db;">🧭 Navigation</h2>
    </div>
""", unsafe_allow_html=True)

app_mode = st.sidebar.radio(
    "Navigation Menu",
    ["🏥 New Assessment", "📂 History & Growth Trends"],
    label_visibility="collapsed"
)

# --- 3. Database Connection (Set directly in source code) ---
# Replace the placeholders below with your Supabase credentials to enable saving assessments
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"

# Fallback to streamlit secrets if not set above
sb_url = SUPABASE_URL if SUPABASE_URL and SUPABASE_URL != "https://your-project.supabase.co" else st.secrets.get("SUPABASE_URL", "")
sb_key = SUPABASE_KEY if SUPABASE_KEY and SUPABASE_KEY != "your-anon-key" else st.secrets.get("SUPABASE_KEY", "")

# Clean up default placeholders
if sb_url == "https://your-project.supabase.co":
    sb_url = ""
if sb_key == "your-anon-key":
    sb_key = ""

st.sidebar.markdown("---")
st.sidebar.subheader("🔌 Database Connection")

if sb_url and sb_key:
    st.sidebar.success("Connected to Supabase")
else:
    st.sidebar.info("Offline Mode (Saving disabled)")

def get_unique_subject_id():
    """Generates a random 4-character ID (1 letter, 3 numbers) and ensures it is unique in Supabase."""
    import random
    import string
    import requests
    
    # If offline, generate a simple random ID
    if not sb_url or not sb_key:
        l = random.choice(string.ascii_uppercase)
        d = "".join(random.choice(string.digits) for _ in range(3))
        return f"{l}{d}"
        
    headers = {
        "apikey": sb_key,
        "Authorization": f"Bearer {sb_key}",
        "Content-Type": "application/json"
    }
    
    # Try up to 100 times to find a unique ID
    for _ in range(100):
        l = random.choice(string.ascii_uppercase)
        d = "".join(random.choice(string.digits) for _ in range(3))
        candidate_id = f"{l}{d}"
        
        endpoint = f"{sb_url.rstrip('/')}/rest/v1/assessments?subject_id=eq.{candidate_id}&select=subject_id"
        try:
            response = requests.get(endpoint, headers=headers)
            if response.status_code == 200:
                if len(response.json()) == 0:
                    return candidate_id
        except Exception:
            pass
            
    # Fallback if checks failed
    l = random.choice(string.ascii_uppercase)
    d = "".join(random.choice(string.digits) for _ in range(3))
    return f"{l}{d}"

# --- 4. Header ---
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
    if app_mode == "🏥 New Assessment":
        # --- 5. Main Assessment Layout ---
        left_col, right_col = st.columns([1.2, 1], gap="large")

        with left_col:
            st.subheader("📝 Subject Assessment")
            
            with st.expander("1. 📏 Growth Metrics (Current & Past History)", expanded=True):
                if 'generated_subject_id' not in st.session_state:
                    st.session_state.generated_subject_id = get_unique_subject_id()
                
                c_sid1, c_sid2 = st.columns([3, 1])
                with c_sid1:
                    subject_id_input = st.text_input("Subject ID (Auto-generated)", value=st.session_state.generated_subject_id, disabled=True, help="Automatically generated unique ID for this child.")
                with c_sid2:
                    st.write("") # spacing
                    st.write("")
                    if st.button("🔄 New ID", help="Generate a new subject ID for a different child"):
                        st.session_state.generated_subject_id = get_unique_subject_id()
                        st.rerun()
                c1, c2 = st.columns(2)
                with c1: gender = st.selectbox("Gender", ["Male", "Female"])
                with c2: current_age = st.number_input("Age (months)", 0, 59, 24)
                
                c3, c4 = st.columns(2)
                with c3: birth_weight = st.number_input("Birth Weight (kg)", 0.5, 6.0, 3.0, step=0.1)
                with c4: st.write("")

                st.markdown("<b>Current Visit Date & Measurements:</b>", unsafe_allow_html=True)
                import datetime
                c_date, c5, c6 = st.columns([1.2, 1, 1])
                with c_date: current_date = st.date_input("Visit Date", datetime.date.today())
                with c5: current_weight = st.number_input("Current Weight (kg)", min_value=2.0, max_value=30.0, value=9.5, step=0.1)
                with c6: current_height = st.number_input("Current Height (cm)", min_value=45.0, max_value=120.0, value=75.0, step=0.1)

                st.markdown("---")
                st.markdown("<b>📜 Past Visits (Optional for Z-Score Trajectory)</b>", unsafe_allow_html=True)
                
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    include_past1 = st.checkbox("Include Past Visit 1 (Most Recent)", value=False)
                with p_col2:
                    include_past2 = st.checkbox("Include Past Visit 2 (Older)", value=False)
                
                past1_data = None
                past2_data = None
                
                if include_past1:
                    st.markdown("<p style='font-weight:600; color:var(--primary-color); margin-bottom: 2px;'>Past Visit 1 (Most Recent)</p>", unsafe_allow_html=True)
                    pc1, pc2, pc3 = st.columns([1.2, 1, 1])
                    with pc1:
                        past1_date = st.date_input("Visit 1 Date", value=current_date - datetime.timedelta(days=30), key="past1_date")
                    with pc2:
                        past1_weight = st.number_input("Visit 1 Weight (kg)", min_value=2.0, max_value=30.0, value=9.0, step=0.1, key="past1_weight")
                    with pc3:
                        past1_height = st.number_input("Visit 1 Height (cm)", min_value=45.0, max_value=120.0, value=73.0, step=0.1, key="past1_height")
                    
                    days_diff1 = (current_date - past1_date).days
                    past1_age = int(round(current_age - (days_diff1 / 30.4375)))
                    if past1_age < 0:
                        st.error("❌ Visit 1 date is before child's birth (age < 0).")
                    elif past1_age > 59:
                        st.error("❌ Visit 1 age exceeds 59 months.")
                    else:
                        st.caption(f"Estimated Age at Visit 1: **{past1_age} months**")
                        past1_data = {"date": past1_date, "weight": past1_weight, "height": past1_height, "age_months": past1_age}
                
                if include_past2:
                    st.markdown("<p style='font-weight:600; color:var(--primary-color); margin-bottom: 2px;'>Past Visit 2 (Older)</p>", unsafe_allow_html=True)
                    pc4, pc5, pc6 = st.columns([1.2, 1, 1])
                    with pc4:
                        past2_date = st.date_input("Visit 2 Date", value=current_date - datetime.timedelta(days=60), key="past2_date")
                    with pc5:
                        past2_weight = st.number_input("Visit 2 Weight (kg)", min_value=2.0, max_value=30.0, value=8.5, step=0.1, key="past2_weight")
                    with pc6:
                        past2_height = st.number_input("Visit 2 Height (cm)", min_value=45.0, max_value=120.0, value=71.0, step=0.1, key="past2_height")
                    
                    days_diff2 = (current_date - past2_date).days
                    past2_age = int(round(current_age - (days_diff2 / 30.4375)))
                    if past2_age < 0:
                        st.error("❌ Visit 2 date is before child's birth (age < 0).")
                    elif past2_age > 59:
                        st.error("❌ Visit 2 age exceeds 59 months.")
                    else:
                        st.caption(f"Estimated Age at Visit 2: **{past2_age} months**")
                        past2_data = {"date": past2_date, "weight": past2_weight, "height": past2_height, "age_months": past2_age}

            with st.expander("2. 💊 Health & Clinical History", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    imm_options = ["Age Appropriate", "Partially Immunized", "Zero Dose"]
                    if current_age >= 12:
                        imm_options.insert(1, "Fully Immunized (12+ months)")
                    
                    imm_label = st.selectbox("Immunization Status", imm_options, index=0)
                    
                    imm_map = {
                        "Age Appropriate": "age_appropriate",
                        "Fully Immunized (12+ months)": "fully_immunized",
                        "Partially Immunized": "partially_immunized",
                        "Zero Dose": "zero_dose"
                    }
                    immunization_status = imm_map[imm_label]

                    recurrent_diarrhea = st.selectbox("Recurrent Diarrhea?", ["yes", "no"], index=1)
                    chronic_illness = st.selectbox("Chronic Illness? (CHD, TB, CP)", ["yes", "no"], index=1)
                with col_b:
                    hiv_options = ["HIV Unexposed", "HIV Exposed Unaffected", "HIV Infected", "Unknown"]
                    hiv_label = st.selectbox("HIV Status", hiv_options, index=0)
                    
                    hiv_map = {
                        "HIV Unexposed": "hiv_unexposed",
                        "HIV Exposed Unaffected": "hiv_exposed_unaffected",
                        "HIV Infected": "hiv_infected",
                        "Unknown": "unknown"
                    }
                    hiv_exposure = hiv_map[hiv_label]
                    
                    recent_illness = st.selectbox("Recent Illness? (in last month)", ["yes", "no"], index=1)

            with st.expander("3. 🍲 Feeding Practices", expanded=True):
                col_c, col_d = st.columns(2)
                with col_c:
                    ebf_duration = st.number_input("Duration of Exclusive Breastfeeding (months)", min_value=0, max_value=12, value=6, help="How many months was the child exclusively breastfed?")
                    breastfeeding_6m = "yes" if ebf_duration >= 6 else "no"
                    meal_freq = 3 # Default value for model compatibility
                with col_d:
                    if current_age >= 6:
                        st.markdown('<p style="color:var(--text-color); font-weight:600;">Complementary Feeding (Select all consumed yesterday):</p>', unsafe_allow_html=True)
                        
                        nutrient_options = [
                            "Breast Milk",
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
                        st.caption(f"Calculated Diversity Score: **{feeding_diversity}/8**")
                    else:
                        st.info(f"ℹ️ Complementary feeding is recommended starting at 6 months. (Current Age: {current_age}m)")
                        feeding_diversity = 0
                        selected_nutrients = []

            with st.expander("4. 🏠 Socio-Economic Factors (SES Score)", expanded=True):
                col_e, col_f = st.columns(2)
                
                with col_e:
                    income_val = st.number_input("Household Monthly Income (KES)", min_value=0, value=5000, step=500, help="Enter monthly household income in Kenyan Shillings.")
                    if income_val < 3000:
                        income_score = 0
                        income_level = "low"
                    elif income_val <= 10000:
                        income_score = 1
                        income_level = "middle"
                    else:
                        income_score = 2
                        income_level = "high"

                with col_f:
                    crowding_val = st.number_input("Household Crowding (Persons per Room)", min_value=1, value=3, step=1, help="Enter the number of persons per room.")
                    if crowding_val < 3:
                        crowding_score = 2
                    elif crowding_val <= 5:
                        crowding_score = 1
                    else:
                        crowding_score = 0

                ses_score_total = income_score + crowding_score
                
                if ses_score_total <= 1:
                    ses_category_label = "Low SES (High Risk)"
                    ses_color = "#e74c3c"
                elif ses_score_total <= 3:
                    ses_category_label = "Middle SES (Moderate Risk)"
                    ses_color = "#f39c12"
                else:
                    ses_category_label = "High SES (Low Risk)"
                    ses_color = "#2ecc71"

                st.markdown("---")
                c_score, c_cat = st.columns([1, 2])
                with c_score:
                    st.markdown(f"**Total SES Score:** <span style='font-size:1.2em'>{ses_score_total} / 4</span>", unsafe_allow_html=True)
                with c_cat:
                    st.markdown(f"**SES Category:** <span style='color:{ses_color}; font-weight:bold; font-size:1.2em'>{ses_category_label}</span>", unsafe_allow_html=True)
                
                # Derive proxy values for features not in the UI
                education_level = "primary" if income_level == "low" else "secondary" if income_level == "middle" else "tertiary"
                water_access = "no" if income_level == "low" else "yes"
                sanitation_access = "no" if (income_level == "low" or crowding_score == 0) else "yes"
                
            st.markdown("<br>", unsafe_allow_html=True)
            age_invalid = current_age < 0 or current_age > 59
            
            # Past visits validation
            date_invalid = False
            date_error_msg = ""
            if include_past1:
                if past1_date >= current_date:
                    date_invalid = True
                    date_error_msg = "❌ **Invalid Dates**: Past Visit 1 date must be before the current visit date."
                elif past1_age < 0 or past1_age > 59:
                    date_invalid = True
                    date_error_msg = "❌ **Invalid Age**: Estimated age at Past Visit 1 must be between 0 and 59 months."
            if include_past2:
                if past2_date >= current_date:
                    date_invalid = True
                    date_error_msg = "❌ **Invalid Dates**: Past Visit 2 date must be before the current visit date."
                elif past2_age < 0 or past2_age > 59:
                    date_invalid = True
                    date_error_msg = "❌ **Invalid Age**: Estimated age at Past Visit 2 must be between 0 and 59 months."
                if include_past1 and past2_date >= past1_date:
                    date_invalid = True
                    date_error_msg = "❌ **Invalid Dates**: Past Visit 2 date must be before Past Visit 1 date."

            if age_invalid:
                st.error("❌ **Invalid Age**: Subject age must be between 0 and 59 months.")
            elif date_invalid:
                st.error(date_error_msg)
                
            predict_btn = st.button("🔍 ANALYZE COMPREHENSIVE RISK", type="primary", disabled=(age_invalid or date_invalid))

        if 'assessment_results' not in st.session_state:
            st.session_state.assessment_results = None

        if predict_btn:
            current_whz, whz_category = calculate_whz(current_height, current_weight, gender)
            z_score = current_whz

            # Build trajectory dataset (Current + Past visits)
            trajectory_list = []
            
            # Current visit
            trajectory_list.append({
                "age_months": int(current_age),
                "weight": float(current_weight),
                "height": float(current_height),
                "z_score": float(z_score) if z_score is not None else None,
                "whz_category": whz_category,
                "visit_label": "Current"
            })
            
            # Past Visit 1
            if include_past1:
                past1_whz, past1_cat = calculate_whz(past1_height, past1_weight, gender)
                trajectory_list.append({
                    "age_months": int(past1_age),
                    "weight": float(past1_weight),
                    "height": float(past1_height),
                    "z_score": float(past1_whz) if past1_whz is not None else None,
                    "whz_category": past1_cat,
                    "visit_label": "Past 1"
                })
                
            # Past Visit 2
            if include_past2:
                past2_whz, past2_cat = calculate_whz(past2_height, past2_weight, gender)
                trajectory_list.append({
                    "age_months": int(past2_age),
                    "weight": float(past2_weight),
                    "height": float(past2_height),
                    "z_score": float(past2_whz) if past2_whz is not None else None,
                    "whz_category": past2_cat,
                    "visit_label": "Past 2"
                })
            
            # Sort trajectory list chronologically by age_months
            trajectory_list = sorted(trajectory_list, key=lambda x: x["age_months"])

            # Input Vector for Model (using all features except MUAC)
            input_vector = pd.DataFrame([{
                'age_months': current_age,
                'weight': current_weight,
                'height': current_height,
                'gender': gender,
                'birth_weight': birth_weight,
                'household_income_level': income_level,
                'parent_education_level': education_level,
                'access_to_clean_water': water_access,
                'sanitation_access': sanitation_access,
                'hiv_exposure': hiv_exposure,
                'chronic_illness': chronic_illness,
                'recurrent_diarrhea': recurrent_diarrhea,
                'exclusive_breastfeeding_6m': breastfeeding_6m,
                'immunization_status': immunization_status,
                'feeding_practice': "Mixed Feeding" if feeding_diversity > 3 else "Complementary Feeding" if current_age >= 6 else "Exclusive Breastfeeding",
                'recent_illness': "Fever" if recent_illness == "yes" else "no",
                'z_score': z_score if z_score is not None else 0.0,
            }])

            # Prediction (Multi-class)
            try:
                probabilities = model.predict_proba(input_vector)[0]
                classes = list(model.classes_)
                
                # Enforce Z-score clinical constraints:
                # Malnourished (Moderate/Severe Malnutrition) ONLY if z_score <= -2.0
                # Otherwise, Not Malnourished (High/Moderate/Low Risk)
                if z_score <= -2.0:
                    allowed_classes = ["Moderate Malnutrition", "Severe Malnutrition"]
                else:
                    allowed_classes = ["High Risk", "Moderate Risk", "Low Risk"]
                
                # Filter probabilities to allowed classes
                allowed_probs = []
                for c in allowed_classes:
                    if c in classes:
                        allowed_probs.append((c, probabilities[classes.index(c)]))
                    else:
                        allowed_probs.append((c, 0.0))
                
                # Find the allowed class with the highest probability
                best_class, best_prob = max(allowed_probs, key=lambda x: x[1])
                
                # If all allowed classes have 0 probability (unlikely), fallback to clinical rules
                if best_prob == 0.0:
                    if z_score <= -3.0:
                        predicted_class = "Severe Malnutrition"
                    elif z_score <= -2.0:
                        predicted_class = "Moderate Malnutrition"
                    else:
                        if z_score <= -1.0:
                            predicted_class = "High Risk"
                        else:
                            predicted_class = "Low Risk"
                    prob = 1.0
                else:
                    # Normalize probability among allowed classes
                    sum_probs = sum(p for c, p in allowed_probs)
                    if sum_probs > 0:
                        prob = best_prob / sum_probs
                    else:
                        prob = best_prob
                    predicted_class = best_class
                
                ml_risk = predicted_class
                
                if "Severe" in ml_risk or "High Risk" in ml_risk:
                    ml_class = "status-danger"
                    bar_color = "#e74c3c"
                elif "Moderate Malnutrition" in ml_risk or "Moderate Risk" in ml_risk:
                    ml_class = "status-warning"
                    bar_color = "#ffa502"
                else:
                    ml_class = "status-normal"
                    bar_color = "#2ecc71"
                    
            except Exception as e:
                prob, ml_risk, ml_class, bar_color = 0, "Error", "status-neutral", "#95a5a6"
                st.error(f"Prediction Error: {e}")

            # Pre-calculate risk factors for explanation
            risk_factors = []
            contributing_factors = []
            
            if z_score is not None:
                if z_score <= -3:
                    risk_factors.append("Severe Acute Malnutrition (WHZ ≤ -3 SD) - Recommend immediate treatment.")
                    contributing_factors.append("Critically low Weight-for-Height Z-Score (SAM)")
                elif z_score <= -2:
                    risk_factors.append("Moderate Acute Malnutrition (-3 < WHZ ≤ -2 SD) - Recommend nutritional treatment.")
                    contributing_factors.append("Low Weight-for-Height Z-Score (MAM)")
                elif z_score <= -1:
                    risk_factors.append("High Risk (-2 < WHZ ≤ -1 SD)")
                    contributing_factors.append("Below average Weight-for-Height Z-Score")
                elif z_score <= 1:
                    # Moderate risk (normal range)
                    contributing_factors.append("Weight-for-Height Z-Score (Moderate Risk)")
                elif z_score < 3:
                    # Not Malnourished - Low Risk
                    pass
                else:
                    risk_factors.append("Obese (WHZ ≥ 3 SD) - Recommend obesity management.")
                    contributing_factors.append("Extremely high Weight-for-Height Z-Score")
            
            if birth_weight < 2.5:
                risk_factors.append("⚠️ Low Birth Weight (< 2.5 kg): Increased Malnutrition Risk")
                contributing_factors.append("Low birth weight (< 2.5 kg)")
                
            if recent_illness == "yes":
                contributing_factors.append("Recent illness history")
                risk_factors.append("Recent Illness (High Risk)")
                
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
            
            if current_age >= 6 and feeding_diversity < 5:
                risk_factors.append("Low Dietary Diversity (< 5 groups)")
                contributing_factors.append("Low feeding diversity score")
            
            if water_access == 'no' or sanitation_access == 'no':
                risk_factors.append("Poor WASH conditions (Infection Risk)")
                contributing_factors.append("Limited access to clean water/sanitation")

            if ses_category_label.startswith("Low SES"):
                risk_factors.append(f"Low Socio-Economic Status (Score: {ses_score_total}/4): High Malnutrition Risk")
                contributing_factors.append("Lower socio-economic factors")

            if len(contributing_factors) == 0:
                contributing_factors.append("No major specific risk factors identified; percentage reflects baseline demographic/growth patterns.")

            # --- Multi-parametric Clinical Assessment Synthesis ---
            growth_score = 0
            growth_recs = []
            
            # Growth metrics
            if z_score is not None:
                if z_score <= -3:
                    growth_score += 3
                    growth_recs.append("Severe Acute Malnutrition (WHZ ≤ -3 SD)")
                elif z_score <= -2:
                    growth_score += 2
                    growth_recs.append("Moderate Wasting (WHZ ≤ -2 SD)")
                elif z_score <= -1:
                    growth_score += 1
                    growth_recs.append("Mild Wasting (WHZ ≤ -1 SD)")
                elif z_score >= 3:
                    growth_score += 2
                    growth_recs.append("Obesity (WHZ ≥ 3 SD)")
                else:
                    growth_recs.append("Normal WHZ Score")
            
            if birth_weight < 2.5:
                growth_score += 1
                growth_recs.append("Low Birth Weight (< 2.5 kg)")
                
            # Trajectory
            trajectory_status = "Single visit recorded"
            if len(trajectory_list) > 1:
                first_z = trajectory_list[0]['z_score']
                last_z = trajectory_list[-1]['z_score']
                if first_z is not None and last_z is not None:
                    z_diff = last_z - first_z
                    if z_diff < -0.2:
                        growth_score += 1
                        trajectory_status = f"Declining Trend (dropped {z_diff:.2f} SD)"
                        growth_recs.append("Deteriorating Trajectory")
                    elif z_diff > 0.2:
                        trajectory_status = f"Improving Trend (gained {z_diff:+.2f} SD)"
                    else:
                        trajectory_status = "Stable Trend"

            # Clinical Pillar
            clinical_score = 0
            clinical_recs = []
            if hiv_exposure == 'hiv_infected':
                clinical_score += 3
                clinical_recs.append("HIV Infected")
            elif hiv_exposure == 'hiv_exposed_unaffected':
                clinical_score += 1
                clinical_recs.append("HIV Exposed")
            elif hiv_exposure == 'unknown':
                clinical_recs.append("HIV Status Unknown")
                
            if immunization_status == 'zero_dose':
                clinical_score += 2
                clinical_recs.append("Zero Dose Vaccination")
            elif immunization_status == 'partially_immunized':
                clinical_score += 1
                clinical_recs.append("Partially Immunized")
                
            if chronic_illness == 'yes':
                clinical_score += 1
                clinical_recs.append("Chronic Illness (CHD, TB, CP)")
                
            if recurrent_diarrhea == 'yes':
                clinical_score += 1
                clinical_recs.append("Recurrent Diarrhea")
                
            if recent_illness == 'yes':
                clinical_score += 1
                clinical_recs.append("Recent Illness")

            # Feeding Pillar
            feeding_score = 0
            feeding_recs = []
            if ebf_duration < 2:
                feeding_score += 2
                feeding_recs.append("Suboptimal EBF (< 2 months)")
            elif ebf_duration <= 5:
                feeding_score += 1
                feeding_recs.append("Early EBF Cessation (2-5 months)")
                
            if current_age >= 6:
                if feeding_diversity < 5:
                    feeding_score += 1
                    feeding_recs.append(f"Inadequate Dietary Diversity ({feeding_diversity}/8 food groups)")
                else:
                    feeding_recs.append(f"Adequate Dietary Diversity ({feeding_diversity}/8 food groups)")
            else:
                feeding_recs.append("Age-appropriate Breastfeeding")

            # Socio-Economic Pillar
            ses_pillar_score = 0
            ses_recs = []
            if ses_score_total <= 1:
                ses_pillar_score += 2
                ses_recs.append(f"Low SES ({ses_category_label})")
            elif ses_score_total <= 3:
                ses_pillar_score += 1
                ses_recs.append(f"Middle SES ({ses_category_label})")
            else:
                ses_recs.append(f"High SES ({ses_category_label})")

            # Total score
            total_ivs = growth_score + clinical_score + feeding_score + ses_pillar_score
            
            # Determine Final Risk Status, Box Color, and Recommendations
            if z_score <= -3 or hiv_exposure == 'hiv_infected' or total_ivs >= 6 or (ml_risk is not None and "Severe" in ml_risk):
                final_status = "🔴 CRITICAL CLINICAL RISK"
                rec_class = "rec-critical"
                final_action = "<b>ACTION: Immediate referral to therapeutic feeding program (SAM Clinic), pediatrician referral, and urgent medical investigation.</b>"
            elif z_score <= -2 or (birth_weight < 2.5 and ses_score_total <= 1) or total_ivs >= 4 or (ml_risk is not None and "Moderate Malnutrition" in ml_risk):
                final_status = "🔴 HIGH CLINICAL RISK"
                rec_class = "rec-critical"
                final_action = "<b>ACTION: Referral to supplementary feeding program, active pediatric growth monitoring, and nutritional counseling.</b>"
            elif z_score <= -1 or total_ivs >= 2 or (ml_risk is not None and "High Risk" in ml_risk):
                final_status = "🟠 MODERATE CLINICAL RISK"
                rec_class = "rec-warning"
                final_action = "<b>ACTION: Routine wellness follow-up, dietary diversity counseling, and WASH advice.</b>"
            else:
                final_status = "🟢 LOW CLINICAL RISK"
                rec_class = "rec-success"
                final_action = "<b>ACTION: General child health maintenance and continuation of healthy feeding practices.</b>"

            # Build formatted HTML report content
            rec_text = f"""
            <div style="font-family: 'Inter', sans-serif;">
                <p style="margin: 0 0 10px 0; font-size: 1.1rem; font-weight: bold;">
                    📋 Integrated Clinical Assessment
                </p>
                <hr style="border: 0; border-top: 1px solid var(--metric-border); margin: 10px 0;">
                <p style="margin: 0 0 10px 0; font-size: 1.05rem;">
                    <b>Diagnostic Status:</b> 
                    <span style="font-weight: bold; font-size: 1.1rem;">{final_status}</span>
                </p>
                <p style="margin: 0 0 15px 0; font-size: 0.92rem;">
                    <b>Clinical Vulnerability Score (CVS):</b> <code>{total_ivs} points</code> 
                    (Growth: {growth_score} | Clinical: {clinical_score} | Feeding: {feeding_score} | SES: {ses_pillar_score})
                </p>
                <div style="margin: 15px 0;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; text-align: left; color: inherit;">
                        <thead>
                            <tr style="border-bottom: 2px solid var(--metric-border);">
                                <th style="padding: 6px; width: 35%;">Clinical Pillar</th>
                                <th style="padding: 6px;">Evaluated Findings</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid var(--metric-border);">
                                <td style="padding: 6px; font-weight: bold;">1. 📏 Growth Profile</td>
                                <td style="padding: 6px;">{", ".join(growth_recs) if growth_recs else "No growth deficits"} <br><span style="font-size:0.8rem; color: var(--text-muted);">({trajectory_status})</span></td>
                            </tr>
                            <tr style="border-bottom: 1px solid var(--metric-border);">
                                <td style="padding: 6px; font-weight: bold;">2. 🩺 Clinical & Health</td>
                                <td style="padding: 6px;">{", ".join(clinical_recs) if clinical_recs else "No clinical risks identified"}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid var(--metric-border);">
                                <td style="padding: 6px; font-weight: bold;">3. 🍲 Feeding Practices</td>
                                <td style="padding: 6px;">{", ".join(feeding_recs) if feeding_recs else "Age-appropriate feeding"}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid var(--metric-border);">
                                <td style="padding: 6px; font-weight: bold;">4. 🏡 Socio-Economic</td>
                                <td style="padding: 6px;">{", ".join(ses_recs) if ses_recs else "Low environmental vulnerability"}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <p style="margin: 15px 0 0 0; font-size: 1.0rem; line-height: 1.4;">
                    {final_action}
                </p>
            </div>
            """
                    
            if whz_category in ("Severe Acute Malnutrition", "Obese", "High Risk"):
                whz_risk_class = "status-danger"
            elif whz_category in ("Moderate Acute Malnutrition", "Moderate Risk"):
                whz_risk_class = "status-warning"
            else:
                whz_risk_class = "status-normal"

            st.session_state.assessment_results = {
                'subject_id': subject_id_input,
                'age_months': current_age,
                'gender': gender,
                'birth_weight': birth_weight,
                'weight': current_weight,
                'height': current_height,
                'recent_illness': recent_illness,
                'chronic_illness': chronic_illness,
                'immunization_status': immunization_status,
                'feeding_practice': "Mixed Feeding" if feeding_diversity > 3 else "Complementary Feeding" if current_age >= 6 else "Exclusive Breastfeeding",
                'household_income_level': income_level,
                'parent_education_level': education_level,
                'access_to_clean_water': water_access,
                'sanitation_access': sanitation_access,
                'hiv_exposure': hiv_exposure,
                'recurrent_diarrhea': recurrent_diarrhea,
                'exclusive_breastfeeding_6m': breastfeeding_6m,
                'feeding_diversity_score': feeding_diversity,
                'ses_score': ses_score_total,
                'z_score': z_score,
                'whz_category': whz_category,
                'trajectory_data': trajectory_list,
                'ml_risk': ml_risk,
                'ml_confidence': prob,
                'risk_factors': risk_factors,
                'contributing_factors': contributing_factors,
                'rec_text': rec_text,
                'rec_class': rec_class,
                'whz_risk_class': whz_risk_class,
                'ml_class': ml_class,
                'bar_color': bar_color
            }

            # Auto-save to Supabase on every analysis run
            if sb_url and sb_key:
                import database as db
                
                # Build single-row payload containing current and optional past visits
                payload = {
                    "subject_id": subject_id_input,
                    "created_at": current_date.isoformat(),
                    "age_months": int(current_age),
                    "gender": gender,
                    "birth_weight": float(birth_weight),
                    "weight": float(current_weight),
                    "height": float(current_height),
                    "recent_illness": recent_illness,
                    "chronic_illness": chronic_illness,
                    "immunization_status": immunization_status,
                    "feeding_practice": "Mixed Feeding" if feeding_diversity > 3 else "Complementary Feeding" if current_age >= 6 else "Exclusive Breastfeeding",
                    "household_income_level": income_level,
                    "parent_education_level": education_level,
                    "access_to_clean_water": water_access,
                    "sanitation_access": sanitation_access,
                    "hiv_exposure": hiv_exposure,
                    "recurrent_diarrhea": recurrent_diarrhea,
                    "exclusive_breastfeeding_6m": breastfeeding_6m,
                    "feeding_diversity_score": int(feeding_diversity),
                    "ses_score": int(ses_score_total),
                    "z_score": float(z_score) if z_score is not None else None,
                    "whz_category": whz_category,
                    "ml_risk": ml_risk,
                    "ml_confidence": float(prob),
                    # Past Visit 1 Columns
                    "past1_age_months": int(past1_age) if include_past1 else None,
                    "past1_weight": float(past1_weight) if include_past1 else None,
                    "past1_height": float(past1_height) if include_past1 else None,
                    "past1_z_score": float(calculate_whz(past1_height, past1_weight, gender)[0]) if include_past1 else None,
                    # Past Visit 2 Columns
                    "past2_age_months": int(past2_age) if include_past2 else None,
                    "past2_weight": float(past2_weight) if include_past2 else None,
                    "past2_height": float(past2_height) if include_past2 else None,
                    "past2_z_score": float(calculate_whz(past2_height, past2_weight, gender)[0]) if include_past2 else None,
                }
                
                success_curr, msg_curr = db.save_assessment(sb_url, sb_key, payload)
                if success_curr:
                    st.toast("Assessment saved successfully to Supabase!", icon="💾")
                    # Regenerate a new subject ID for the next assessment
                    st.session_state.generated_subject_id = get_unique_subject_id()
                else:
                    st.error(f"Failed to auto-save record: {msg_curr}")

        with right_col:
            st.subheader("📊 Clinical Assessment Results")
            
            if st.session_state.assessment_results is not None:
                res = st.session_state.assessment_results
                
                r1_col1, r1_col2 = st.columns(2)
                with r1_col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <p class="label-text">🤖 Trend Prediction</p>
                        <div class="status-badge {res['ml_class']}">{res['ml_risk']}</div>
                        <p style="margin-top: 15px; color: var(--text-muted); font-size: 0.9rem;">Confidence: <b style="color: var(--text-color); font-size: 1.2rem; font-weight: 800;">{res['ml_confidence']:.1%}</b></p>
                        <div class="progress-bar-container">
                            <div class="progress-bar-fill" style="width: {res['ml_confidence']*100}%; background-color: {res['bar_color']};"></div>
                          </div>
                      </div>""", unsafe_allow_html=True)
                      
                    st.markdown(f"""<div class="rec-box {res['rec_class']}" style="margin-top: 10px; padding: 15px;">{res['rec_text']}</div>""", unsafe_allow_html=True)
                with r1_col2:
                    whz_val_str = f"{res['z_score']:.2f}" if res['z_score'] is not None else "N/A"
                    whz_status_str = "Malnourished" if res['z_score'] is not None and res['z_score'] <= -2.0 else "Not Malnutritioned"
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <p class="label-text">📏 Weight-for-Height Z-Score</p>
                        <div class="status-badge {res['whz_risk_class']}">{whz_status_str}</div>
                        <p style="margin-top: 15px; color: var(--text-muted); font-size: 0.9rem;">Value: <b style="color: var(--text-color); font-size: 1.2rem; font-weight: 800;">{whz_val_str} SD</b></p>
                    </div>""", unsafe_allow_html=True)
                
                # --- 6. Trajectory of the Z-Score ---
                st.markdown("---")
                st.markdown("### 📈 Weight-for-Height Z-Score Trajectory")
                
                # Fetch all records for this subject from Supabase to show full history, if connected
                db_records = []
                if sb_url and sb_key:
                    try:
                        import database as db
                        db_records = db.get_assessments_by_subject(sb_url, sb_key, res['subject_id'])
                    except Exception as e:
                        pass
                
                if db_records:
                    latest_rec = db_records[-1] # Ordered by id ASC, so the last is latest
                    trajectory_list = []
                    
                    # Current
                    trajectory_list.append({
                        "age_months": int(latest_rec['age_months']),
                        "weight": float(latest_rec['weight']),
                        "height": float(latest_rec['height']),
                        "z_score": float(latest_rec['z_score']) if latest_rec.get('z_score') is not None else None,
                        "whz_category": latest_rec.get('whz_category', 'Normal'),
                        "visit_label": "Current"
                    })
                    
                    # Past Visit 1
                    if latest_rec.get('past1_age_months') is not None:
                        past1_whz = latest_rec.get('past1_z_score')
                        _, past1_cat = calculate_whz(latest_rec['past1_height'], latest_rec['past1_weight'], latest_rec['gender'])
                        trajectory_list.append({
                            "age_months": int(latest_rec['past1_age_months']),
                            "weight": float(latest_rec['past1_weight']),
                            "height": float(latest_rec['past1_height']),
                            "z_score": float(past1_whz) if past1_whz is not None else None,
                            "whz_category": past1_cat,
                            "visit_label": "Past 1"
                        })
                        
                    # Past Visit 2
                    if latest_rec.get('past2_age_months') is not None:
                        past2_whz = latest_rec.get('past2_z_score')
                        _, past2_cat = calculate_whz(latest_rec['past2_height'], latest_rec['past2_weight'], latest_rec['gender'])
                        trajectory_list.append({
                            "age_months": int(latest_rec['past2_age_months']),
                            "weight": float(latest_rec['past2_weight']),
                            "height": float(latest_rec['past2_height']),
                            "z_score": float(past2_whz) if past2_whz is not None else None,
                            "whz_category": past2_cat,
                            "visit_label": "Past 2"
                        })
                        
                    trajectory_df = pd.DataFrame(trajectory_list)
                    trajectory_df = trajectory_df.sort_values(by="age_months").drop_duplicates(subset=["age_months"])
                else:
                    trajectory_df = pd.DataFrame(res.get('trajectory_data', []))
                    if not trajectory_df.empty:
                        trajectory_df = trajectory_df.sort_values(by="age_months").drop_duplicates(subset=["age_months"])
                
                if not trajectory_df.empty and len(trajectory_df) > 1:
                    # Plot Z-score trajectory with critical regions (interactive Altair chart)
                    chart = plot_whz_trajectory_with_regions(trajectory_df)
                    st.altair_chart(chart, use_container_width=True)
                    
                    # Clinical trajectory assessment
                    # Sort trajectory_df to ensure first and last visits are chronologically correct
                    trajectory_df_sorted = trajectory_df.sort_values(by="age_months")
                    first_visit = trajectory_df_sorted.iloc[0]
                    last_visit = trajectory_df_sorted.iloc[-1]
                    z_diff = last_visit['z_score'] - first_visit['z_score']
                    
                    if z_diff > 0.2:
                        traj_status = "Improving 📈"
                        traj_class = "status-normal"
                        traj_note = f"The child's WHZ has improved by **+{z_diff:.2f} SD** (from {first_visit['z_score']:.2f} SD to {last_visit['z_score']:.2f} SD)."
                    elif z_diff < -0.2:
                        traj_status = "Deteriorating 📉"
                        traj_class = "status-danger"
                        traj_note = f"The child's WHZ has declined by **{z_diff:.2f} SD** (from {first_visit['z_score']:.2f} SD to {last_visit['z_score']:.2f} SD). Immediate intervention is advised."
                    else:
                        traj_status = "Stable ➡️"
                        traj_class = "status-warning"
                        traj_note = f"The child's growth is stable with a WHZ change of **{z_diff:+.2f} SD** (current: {last_visit['z_score']:.2f} SD)."
                        
                    st.markdown(f"""
                    <div style="background-color: var(--card-bg); border: 1px solid var(--metric-border); padding: 15px; border-radius: 8px; margin-top: 10px;">
                        <p style="margin: 0; font-size: 0.95rem; line-height: 1.5;">
                            <b>Trajectory Status:</b> <span class="status-badge {traj_class}" style="margin: 0 0 0 5px; padding: 2px 10px; font-size: 0.85rem;">{traj_status}</span><br><br>
                            {traj_note}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("ℹ&nbsp; Trajectory chart requires at least two visits. Add past visits or run subsequent assessments for this subject to view trajectory.")
                
                # Auto-saved status indicator
                if sb_url and sb_key:
                    st.markdown("---")
                    st.success("✅ Assessment automatically saved to Supabase database.")
                else:
                    st.markdown("---")
                    st.info("💡 Connect to Supabase in the sidebar to enable automatic assessment saving.")
            else:
                st.info("👈 Please enter the current measurements and click Analyze.")

    elif app_mode == "📂 History & Growth Trends":
        st.subheader("📂 Historical Records & Growth Trends")
        
        if not sb_url or not sb_key:
            st.warning("⚠️ Database connection is not configured. Please enter your Supabase URL and Key in the sidebar to view history.")
        else:
            with st.spinner("Fetching assessment records from Supabase..."):
                import database as db
                all_records = db.get_all_assessments(sb_url, sb_key)
                
            if not all_records:
                st.info("ℹ️ No assessment records found in the database. Perform an assessment and save it to see historical data.")
            else:
                # Convert records to DataFrame
                df_records = pd.DataFrame(all_records)
                
                # Format timestamps/dates nicely
                if "created_at" in df_records.columns:
                    df_records["created_at"] = pd.to_datetime(df_records["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                
                st.markdown("### 🔍 Subject Selection")
                unique_subjects = sorted(list(df_records["subject_id"].unique()))
                selected_subject = st.selectbox("Select Subject (Subject ID):", ["-- Show All Records --"] + unique_subjects)
                
                if selected_subject == "-- Show All Records --":
                    st.markdown("#### All Assessments")
                    display_cols = [col for col in ["subject_id", "created_at", "age_months", "gender", "weight", "height", "z_score", "whz_category", "ml_risk", "ml_confidence"] if col in df_records.columns]
                    st.dataframe(df_records[display_cols], use_container_width=True)
                else:
                    subject_data = df_records[df_records["subject_id"] == selected_subject].sort_values(by="age_months")
                    
                    st.markdown(f"### 📊 Historical Trends for Subject: **{selected_subject}**")
                    
                    latest_rec = subject_data.iloc[-1]
                    
                    c_age, c_weight, c_height, c_whz = st.columns(4)
                    c_age.metric("Current Age", f"{latest_rec['age_months']} months")
                    c_weight.metric("Current Weight", f"{latest_rec['weight']} kg")
                    c_height.metric("Current Height", f"{latest_rec['height']} cm")
                    
                    z_val = latest_rec['z_score']
                    z_str = f"{z_val:.2f} SD" if z_val is not None else "N/A"
                    whz_status_hist = "Malnourished" if z_val is not None and z_val <= -2.0 else "Not Malnutritioned"
                    c_whz.metric("Weight-for-Height Z-Score", z_str, delta=whz_status_hist)
                    
                    # Graphing columns
                    st.markdown("#### Growth Trajectories")
                    chart_col1, chart_col2 = st.columns(2)
                    
                    # Rebuild the full trajectory from the latest record's columns to support single-row past visits
                    hist_traj_list = []
                    # Current
                    hist_traj_list.append({
                        "age_months": int(latest_rec['age_months']),
                        "weight": float(latest_rec['weight']),
                        "height": float(latest_rec['height']),
                        "z_score": float(latest_rec['z_score']) if latest_rec.get('z_score') is not None else None,
                        "whz_category": latest_rec.get('whz_category', 'Normal'),
                        "visit_label": "Current"
                    })
                    # Past 1
                    if 'past1_age_months' in latest_rec and pd.notna(latest_rec['past1_age_months']):
                        past1_whz = latest_rec.get('past1_z_score')
                        _, past1_cat = calculate_whz(latest_rec['past1_height'], latest_rec['past1_weight'], latest_rec['gender'])
                        hist_traj_list.append({
                            "age_months": int(latest_rec['past1_age_months']),
                            "weight": float(latest_rec['past1_weight']),
                            "height": float(latest_rec['past1_height']),
                            "z_score": float(past1_whz) if past1_whz is not None else None,
                            "whz_category": past1_cat,
                            "visit_label": "Past 1"
                        })
                    # Past 2
                    if 'past2_age_months' in latest_rec and pd.notna(latest_rec['past2_age_months']):
                        past2_whz = latest_rec.get('past2_z_score')
                        _, past2_cat = calculate_whz(latest_rec['past2_height'], latest_rec['past2_weight'], latest_rec['gender'])
                        hist_traj_list.append({
                            "age_months": int(latest_rec['past2_age_months']),
                            "weight": float(latest_rec['past2_weight']),
                            "height": float(latest_rec['past2_height']),
                            "z_score": float(past2_whz) if past2_whz is not None else None,
                            "whz_category": past2_cat,
                            "visit_label": "Past 2"
                        })
                    
                    hist_traj_df = pd.DataFrame(hist_traj_list).sort_values(by="age_months").drop_duplicates(subset=["age_months"])
                    
                    with chart_col1:
                        st.markdown("<p style='font-weight:600; text-align:center;'>⚖️ Weight Trajectory (kg)</p>", unsafe_allow_html=True)
                        st.line_chart(data=hist_traj_df, x="age_months", y="weight")
                        
                    with chart_col2:
                        st.markdown("<p style='font-weight:600; text-align:center;'>📏 Z-Score (WHZ) Trajectory (SD)</p>", unsafe_allow_html=True)
                        if len(hist_traj_df) > 1:
                            chart_hist = plot_whz_trajectory_with_regions(hist_traj_df)
                            st.altair_chart(chart_hist, use_container_width=True)
                        else:
                            st.info("ℹ️ Trajectory chart requires at least two visits.")
                        
                    # Detailed History Table for this child
                    st.markdown("#### Visit Details")
                    detail_cols = [col for col in ["created_at", "age_months", "weight", "height", "z_score", "whz_category", "ml_risk", "ml_confidence", "feeding_practice", "recent_illness", "chronic_illness"] if col in subject_data.columns]
                    st.dataframe(subject_data[detail_cols], use_container_width=True)
