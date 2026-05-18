import streamlit as st
import requests

st.set_page_config(page_title="AI Salary Predictor", page_icon="💼", layout="centered")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%);
    }
    .hero {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #22c55e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #9ca3af;
        font-size: 1.05rem;
    }
    .section-card {
        background: #1f2937;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.1rem;
        border: 1px solid #374151;
    }
    .section-title {
        color: #d1d5db;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 1rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .result-card {
        text-align: center;
        padding: 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 1px solid #22c55e55;
        box-shadow: 0 0 40px #22c55e18;
        margin-top: 1rem;
    }
    .result-amount {
        font-size: 3.2rem;
        font-weight: 800;
        color: #22c55e;
        letter-spacing: -1px;
    }
    .result-label {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-top: 0.4rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1, #22c55e);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .footer {
        text-align: center;
        color: #4b5563;
        font-size: 0.82rem;
        padding: 1rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>💼 AI Salary Predictor</h1>
    <p>Estimate your data science salary based on role, experience & location</p>
</div>
""", unsafe_allow_html=True)

# --- Mappings ---
experience_map = {
    "EN — Entry-Level": "EN",
    "MI — Mid-Level": "MI",
    "SE — Senior": "SE",
    "EX — Executive / Director": "EX",
}

employment_map = {
    "FT — Full-Time": "FT",
    "PT — Part-Time": "PT",
    "CT — Contract": "CT",
    "FL — Freelance": "FL",
}

company_size_map = {
    "S — Small  (<50 employees)": "S",
    "M — Medium  (50–250 employees)": "M",
    "L — Large  (250+ employees)": "L",
}

country_map = {
    "🇺🇸 US — United States": "US",
    "🇮🇳 IN — India": "IN",
    "🇬🇧 GB — United Kingdom": "GB",
    "🇨🇦 CA — Canada": "CA",
    "🇩🇪 DE — Germany": "DE",
    "🇫🇷 FR — France": "FR",
    "🇯🇵 JP — Japan": "JP",
}

# --- Section 1: Role & Experience ---
st.markdown('<div class="section-card"><div class="section-title">🎯 Role & Experience</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    work_year = st.number_input("Work Year", min_value=2020, max_value=2030, value=2023)
    experience_label = st.selectbox("Experience Level", list(experience_map.keys()))
with col2:
    job_title = st.selectbox("Job Title", ["Data Scientist", "Machine Learning Engineer", "Data Analyst"])
    employment_label = st.selectbox("Employment Type", list(employment_map.keys()))
st.markdown('</div>', unsafe_allow_html=True)

# --- Section 2: Location ---
st.markdown('<div class="section-card"><div class="section-title">🌍 Location</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    residence_label = st.selectbox("Employee Residence", list(country_map.keys()))
with col4:
    company_location_label = st.selectbox("Company Location", list(country_map.keys()))
st.markdown('</div>', unsafe_allow_html=True)

# --- Section 3: Work Setup ---
st.markdown('<div class="section-card"><div class="section-title">🏢 Work Setup</div>', unsafe_allow_html=True)
remote_ratio = st.slider("Remote Ratio", 0, 100, 50, format="%d%%")
company_size_label = st.selectbox("Company Size", list(company_size_map.keys()))
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Build payload ---
data = {
    "work_year": work_year,
    "experience_level": experience_map[experience_label],
    "employment_type": employment_map[employment_label],
    "job_title": job_title,
    "employee_residence": country_map[residence_label],
    "remote_ratio": remote_ratio,
    "company_location": country_map[company_location_label],
    "company_size": company_size_map[company_size_label],
}

# --- Predict ---
if st.button("🚀 Predict My Salary"):
    with st.spinner("Crunching numbers..."):
        try:
            response = requests.post(
                "https://salary-api-9f1v.onrender.com/predict",
                json=data
            )
            result = response.json()

            if "predicted_salary" in result:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-amount">${result['predicted_salary']:,.0f}</div>
                    <div class="result-label">Estimated Annual Salary (USD)</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(result.get("error", "Something went wrong."))
        except Exception:
            st.error("⚠️ Could not reach the API. Please try again.")

st.markdown("---")
st.markdown("<div class='footer'>Built with ML · FastAPI · Streamlit</div>", unsafe_allow_html=True)
