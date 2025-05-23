from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import logging
import mysql.connector
from mysql.connector import Error
from typing import Optional
from dotenv import load_dotenv
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables FIRST
load_dotenv("/etc/secrets/db.env")

# Define the directory where models are saved (relative to this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

dose_ranges = {
    'amlodipine': (5, 10),
    'telmisartan': (40, 80),
    'losartan': (50, 100),
    'metoprolol succinate': (25, 100),
    'furosemide': (40, 80),
    'hydrochlorothiazide': (25, 100),
    'none': (0, 0)
}

def map_dose(medication, category):
    medication = medication.strip().lower()
    if medication in dose_ranges:
        min_dose, max_dose = dose_ranges[medication]
        if category == 'Low':
            return min_dose
        elif category == 'Medium':
            return (min_dose + max_dose) / 2
        elif category == 'High':
            return max_dose
    return 0

# Load models and encoders
model_medication = joblib.load(os.path.join(MODEL_DIR, 'medication_model.pkl'))
medication_encoder = joblib.load(os.path.join(MODEL_DIR, 'medication_encoder.pkl'))
label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
label_encoder_medication = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_medication.pkl'))
label_encoder_dose = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_dose.pkl'))
model_drug_class = joblib.load(os.path.join(MODEL_DIR, 'drug_class_model.pkl'))
model_dose = joblib.load(os.path.join(MODEL_DIR, 'dose_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

class InputData(BaseModel):
    Age: float
    Glucose: float
    Pulse: float
    Diastolic_BP: float
    Systolic_BP: float
    BMI: float
    Creatinine_mg_dl: float
    Medication1_Name: str
    Medication2_Name: str
    Medication3_Name: str
    country: str
    patient_id: str
    submitted_by_token: Optional[str] = None

app = FastAPI()

# Define allowed origins for CORS (Render and local development)
origins = [
    "https://hypertension-form.onrender.com",
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:5000"
]

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "API is working"}

@app.middleware("http")
async def log_request(request: Request, call_next):
    try:
        body = await request.body()
        logger.info(f"Raw request body: {body.decode('utf-8')}")
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Error in middleware: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/predict")
async def predict(data: InputData):
    # Initialize fresh database connection for each request
    try:
        db_connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            port=int(os.getenv("DB_PORT", 3306))  # fallback to 3306 if not set          
        )
        cursor = db_connection.cursor()
    except Error as db_conn_error:
        logger.error(f"Database connection failed: {str(db_conn_error)}")
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        logger.info(f"Incoming request data: {data.dict()}")

        if data.Systolic_BP < 140 and data.Diastolic_BP < 90:
            return {
                "message": "No medication needed. Blood pressure is within normal range.",
                "medication": "None",
                "drug_class": "None",
                "dose_value": 0
            }

        country = data.country
        patient_id = data.patient_id
        logger.info(f"Patient Profile - Country: {country}, Patient ID: {patient_id}")

        creatinine_bmi_ratio = data.Creatinine_mg_dl / data.BMI
        map_value = (data.Systolic_BP + 2 * data.Diastolic_BP) / 3

        input_dict = data.dict()
        input_dict["Creatinine_BMI_Ratio"] = creatinine_bmi_ratio
        input_dict["MAP"] = map_value

        input_df = pd.DataFrame([input_dict])
        logger.info(f"DataFrame after conversion:\n{input_df}")

        input_df['MAP'] = map_value
        input_df['Creatinine_BMI_Ratio'] = creatinine_bmi_ratio

        input_df.rename(columns={
            "Diastolic_BP": "Diastolic BP",
            "Systolic_BP": "Systolic BP",
            "Creatinine_BMI_Ratio": "Creatinine_BMI_Ratio",
            "Medication1_Name": "Medication1 Name",
            "Medication2_Name": "Medication2 Name",
            "Medication3_Name": "Medication3 Name"
        }, inplace=True)

        logger.info(f"DataFrame after renaming columns:\n{input_df}")

        medication_columns = ['Medication1 Name', 'Medication2 Name', 'Medication3 Name']
        medication_encoded = medication_encoder.transform(input_df[medication_columns])
        medication_encoded_df = pd.DataFrame(medication_encoded, 
                                          columns=medication_encoder.get_feature_names_out(medication_columns))

        logger.info(f"Encoded medication DataFrame:\n{medication_encoded_df}")

        numerical_features = ['Age', 'Glucose', 'Pulse', 'MAP', 'Diastolic BP', 'Systolic BP', 'Creatinine_BMI_Ratio']
        X = pd.concat([input_df[numerical_features], medication_encoded_df], axis=1)

        logger.info(f"Combined feature DataFrame:\n{X}")

        X_scaled = scaler.transform(X)
        logger.info(f"Scaled feature matrix:\n{X_scaled}")

        if X_scaled.shape[1] != 24:
            raise ValueError(f"Expected 24 features, but got {X_scaled.shape[1]}")

        dose_pred = model_dose.predict(X_scaled)
        drug_class_pred = model_drug_class.predict(X_scaled)
        medication_pred = model_medication.predict(X_scaled)

        dose_category = label_encoder_dose.inverse_transform(dose_pred)[0]
        drug_class = label_encoder.inverse_transform(drug_class_pred)[0]
        medication = label_encoder_medication.inverse_transform(medication_pred)[0]

        predicted_dose_value = map_dose(medication, dose_category)
        logger.info(f"Predictions - Medication: {medication}, Drug Class: {drug_class}, Dose Value: {predicted_dose_value}")

        # Database insertion
        try:
            cursor.execute("""
                INSERT INTO hypertension_predictions (
                    age, glucose, pulse, systolic_bp, diastolic_bp, bmi, creatinine,
                    creatinine_bmi_ratio, predicted_drug_class, predicted_dose_value,
                    predicted_medication_name, country, patient_id, submitted_by_token
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                data.Age, data.Glucose, data.Pulse, data.Systolic_BP, data.Diastolic_BP,
                data.BMI, data.Creatinine_mg_dl, creatinine_bmi_ratio, drug_class,
                predicted_dose_value, medication, data.country, data.patient_id, data.submitted_by_token
            ))
            db_connection.commit()
            record_id = cursor.lastrowid  # Add this line
            logger.info("Successfully saved prediction to database")
        except Error as db_error:
            logger.error(f"Database save failed: {str(db_error)}")
            # Don't fail the request just because DB save failed
            # Continue to return predictions

        return {
            "medication": medication,
            "drug_class": drug_class,
            "dose_value": predicted_dose_value,
            "id": record_id  # ← now frontend can store and use it
        }

    except ValueError as ve:
        logger.error(f"Value error during prediction: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        try:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'db_connection' in locals() and db_connection:
                db_connection.close()
        except Error as e:
            logger.error(f"Error closing database resources: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
