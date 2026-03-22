from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load trained model
model = joblib.load("model/salary_model.pkl")
encoders = joblib.load("model/encoders.pkl") 

@app.get("/")
def home():
    return {"message": "Salary Predictor API is running 🚀"}


@app.post("/predict")
def predict(data: dict):
    try:
        # Encode using saved encoders
        for col in encoders:
            if col in data:
                le = encoders[col]
                data[col] = le.transform([data[col]])[0]

        # ✅ FIXED: enforce correct feature order
        feature_order = [
            "work_year",
            "experience_level",
            "employment_type",
            "job_title",
            "employee_residence",
            "remote_ratio",
            "company_location",
            "company_size"
        ]

        features = np.array([[data[col] for col in feature_order]])

        prediction = model.predict(features)[0]

        return {"predicted_salary": round(float(prediction), 2)}

    except Exception as e:
        return {"error": str(e)}