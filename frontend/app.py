import streamlit as st
import requests

# Page config
st.set_page_config(page_title="AI Salary Predictor", page_icon="💼", layout="centered")

# Header
st.markdown(
    """
    <h1 style='text-align: center;'>💼 AI Salary Predictor</h1>
    <p style='text-align: center; color: gray;'>Predict salaries based on job role, experience & company details</p>
    """,
    unsafe_allow_html=True
)

# Input Section
with st.container():
    st.subheader("📋 Job Details")

    col1, col2 = st.columns(2)

    with col1:
        work_year = st.number_input("Work Year", value=2023)
        experience_level = st.selectbox("Experience Level", ["EN", "MI", "SE", "EX"])
        job_title = st.selectbox(
            "Job Title",
            ["Data Scientist", "Machine Learning Engineer", "Data Analyst"]
        )

    with col2:
        employment_type = st.selectbox("Employment Type", ["FT", "PT", "CT", "FL"])
        employee_residence = st.selectbox(
            "Employee Residence",
            ["US", "IN", "GB", "CA", "DE", "FR", "JP"]
        )
        company_location = st.selectbox(
            "Company Location",
            ["US", "IN", "GB", "CA", "DE", "FR", "JP"]
        )

# Additional Inputs
remote_ratio = st.slider("🌍 Remote Ratio (%)", 0, 100, 50)
company_size = st.selectbox("🏢 Company Size", ["S", "M", "L"])

# Prepare data (NO encoding here)
data = {
    "work_year": work_year,
    "experience_level": experience_level,
    "employment_type": employment_type,
    "job_title": job_title,
    "employee_residence": employee_residence,
    "remote_ratio": remote_ratio,
    "company_location": company_location,
    "company_size": company_size
}

st.markdown("---")

# Prediction Button
if st.button("🚀 Predict Salary"):
    try:
        with st.spinner("Predicting salary..."):
            response = requests.post(
                "https://salary-api-9f1v.onrender.com/predict",
                json=data
            )

        result = response.json()

        if "predicted_salary" in result:
            st.markdown(
                f"""
                <div style='text-align:center; padding:25px; border-radius:12px; background-color:#1f2937;'>
                    <h2 style='color:#22c55e;'>💰 ${result['predicted_salary']:.2f}</h2>
                    <p style='color:gray;'>Estimated Annual Salary</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error(result.get("error", "Unknown error"))

    except:
        st.error("⚠️ API not running. Start FastAPI first!")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with ❤️ using ML + FastAPI + Streamlit</p>",
    unsafe_allow_html=True
)