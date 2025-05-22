 ####Correction_api with single and combo_prescriptions

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mysql.connector
from typing import List, Optional
from dotenv import load_dotenv
import os

# Load environment variables from secret mount path
load_dotenv("/etc/secrets/db.env")

app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Correction API is running"}

# Initialize MySQL connection using environment variables
db_connection = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
    port=int(os.getenv("DB_PORT", 3306))  # Default to 3306 if not provided
)

# Define models for both single and combo corrections
class SingleCorrectionData(BaseModel):
    patient_id: str
    id: int
    accept_prescription: bool
    corrected_drug_class: Optional[str] = None
    corrected_medication_name: Optional[str] = None
    corrected_dose_value: Optional[float] = None

class ComboCorrectionData(BaseModel):
    patient_id: str
    id: int
    drugs: List[dict]  # Format: [{"class":str, "medication":str, "dose":float}, ...]

@app.post("/correct")
def correct_prediction(correction: SingleCorrectionData):
    """Handle single prescription corrections"""
    try:
        cursor = db_connection.cursor()
        
        # Check if the record exists
        cursor.execute("""
            SELECT id FROM hypertension_predictions
            WHERE id = %s AND patient_id = %s
        """, (correction.id, correction.patient_id))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Record not found for the given patient")

        # Process the correction
        if correction.accept_prescription:
            cursor.execute("""
                UPDATE hypertension_predictions
                SET
                    corrected_drug_class = predicted_drug_class,
                    corrected_medication_name = predicted_medication_name,
                    corrected_dose_value = predicted_dose_value,
                    corrected_drug_class2 = NULL,
                    corrected_medication2 = NULL,
                    corrected_dose_value2 = NULL,
                    corrected_drug_class3 = NULL,
                    corrected_medication3 = NULL,
                    corrected_dose_value3 = NULL
                WHERE id = %s AND patient_id = %s
            """, (correction.id, correction.patient_id))
        else:
            cursor.execute("""
                UPDATE hypertension_predictions
                SET
                    corrected_drug_class = %s,
                    corrected_medication_name = %s,
                    corrected_dose_value = %s,
                    corrected_drug_class2 = NULL,
                    corrected_medication2 = NULL,
                    corrected_dose_value2 = NULL,
                    corrected_drug_class3 = NULL,
                    corrected_medication3 = NULL,
                    corrected_dose_value3 = NULL
                WHERE id = %s AND patient_id = %s
            """, (
                correction.corrected_drug_class,
                correction.corrected_medication_name,
                correction.corrected_dose_value,
                correction.id,
                correction.patient_id
            ))

        db_connection.commit()
        return {"message": "Single correction saved successfully"}

    except mysql.connector.Error as err:
        db_connection.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {err}")

@app.post("/api/correct_combo")
def correct_combo_prediction(correction: ComboCorrectionData):
    """Handle combo prescription corrections"""
    try:
        cursor = db_connection.cursor()
        
        # Check if the record exists
        cursor.execute("""
            SELECT id FROM hypertension_predictions
            WHERE id = %s AND patient_id = %s
        """, (correction.id, correction.patient_id))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Record not found for the given patient")

        # Validate ACEI+ARB contraindication
        classes = [d["class"] for d in correction.drugs if d]
        if ("ACEI" in classes) and ("ARB" in classes):
            raise HTTPException(
                status_code=422,
                detail="ACEI+ARB combination not permitted"
            )

        # Pad with None if <3 drugs
        corrections = correction.drugs + [None] * (3 - len(correction.drugs))

        # Update the database
        cursor.execute("""
            UPDATE hypertension_predictions SET
                corrected_drug_class = %s,
                corrected_medication_name = %s,
                corrected_dose_value = %s,
                corrected_drug_class2 = %s,
                corrected_medication2 = %s,
                corrected_dose_value2 = %s,
                corrected_drug_class3 = %s,
                corrected_medication3 = %s,
                corrected_dose_value3 = %s
            WHERE id = %s AND patient_id = %s
        """, (
            corrections[0]["class"] if corrections[0] else None,
            corrections[0]["medication"] if corrections[0] else None,
            corrections[0]["dose"] if corrections[0] else None,
            corrections[1]["class"] if len(corrections) > 1 and corrections[1] else None,
            corrections[1]["medication"] if len(corrections) > 1 and corrections[1] else None,
            corrections[1]["dose"] if len(corrections) > 1 and corrections[1] else None,
            corrections[2]["class"] if len(corrections) > 2 and corrections[2] else None,
            corrections[2]["medication"] if len(corrections) > 2 and corrections[2] else None,
            corrections[2]["dose"] if len(corrections) > 2 and corrections[2] else None,
            correction.id,
            correction.patient_id
        ))

        db_connection.commit()
        return {"message": "Combo correction saved successfully"}

    except mysql.connector.Error as err:
        db_connection.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {err}")

def get_prediction_values(record_id, patient_id):
    cursor = db_connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT predicted_drug_class, predicted_medication_name, predicted_dose_value
        FROM hypertension_predictions
        WHERE id = %s AND patient_id = %s
    """, (record_id, patient_id))
    return cursor.fetchone()

class AcceptPrescription(BaseModel):
    patient_id: str
    id: int
    accept_prescription: bool

@app.post("/accept_prescription")
def accept_prescription_values(data: AcceptPrescription):
    try:
        predicted = get_prediction_values(data.id, data.patient_id)
        if not predicted:
            raise HTTPException(status_code=404, detail="Prescription not found")

        cursor = db_connection.cursor()
        cursor.execute("""
            UPDATE hypertension_predictions
            SET
                corrected_drug_class = %s,
                corrected_medication_name = %s,
                corrected_dose_value = %s,
                corrected_drug_class2 = NULL,
                corrected_medication2 = NULL,
                corrected_dose_value2 = NULL,
                corrected_drug_class3 = NULL,
                corrected_medication3 = NULL,
                corrected_dose_value3 = NULL
            WHERE id = %s AND patient_id = %s
        """, (
            predicted["predicted_drug_class"],
            predicted["predicted_medication_name"],
            predicted["predicted_dose_value"],
            data.id,
            data.patient_id
        ))

        db_connection.commit()
        return {"message": "Prescription accepted and stored"}

    except mysql.connector.Error as err:
        db_connection.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {err}")
