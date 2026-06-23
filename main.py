from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model
diabetes_model = joblib.load("models/diabetes_model.pkl")
ckd_model = joblib.load("models/ckd_model.pkl")
hyper_model = joblib.load("models/hyperlipidemia_model.pkl")
cad_model = joblib.load("models/cad_model.pkl")
heart_failure_model = joblib.load("models/heart_failure_model.pkl")
obesity_model = joblib.load("models/obesity_model.pkl")



class DiabetesRequest(BaseModel):
    age: float
    bmi: float
    systolic_bp: float
    diastolic_bp: float
    HbA1c: float
    glucose_fasting: float
    sex: int
    smoking_status: int
    exercise_level: int

class CKDRequest(BaseModel):
    age: float
    bmi: float
    systolic_bp: float
    diastolic_bp: float
    creatinine: float
    eGFR: float
    sex: int
    smoking_status: int
    charlson_index: float

class HyperlipidemiaRequest(BaseModel):
    LDL: float
    HDL: float
    triglycerides: float
    total_cholesterol: float
    age: float
    bmi: float    

class CADRequest(BaseModel):
    age: float
    bmi: float
    systolic_bp: float
    diastolic_bp: float
    LDL: float
    HDL: float
    triglycerides: float
    total_cholesterol: float
    BNP: float
    troponin_I: float
    sex: int
    smoking_status: int
    charlson_index: float  


class HeartFailureRequest(BaseModel):
    age: float
    bmi: float
    systolic_bp: float
    diastolic_bp: float
    BNP: float
    troponin_I: float
    charlson_index: float
    sex: int
    smoking_status: int





    


@app.get("/")
def home():
    return {"message": "MedCheck API Running"}


@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesRequest):

    input_df = pd.DataFrame([{
        "age": data.age,
        "bmi": data.bmi,
        "systolic_bp": data.systolic_bp,
        "diastolic_bp": data.diastolic_bp,
        "HbA1c": data.HbA1c,
        "glucose_fasting": data.glucose_fasting,
        "sex": data.sex,
        "smoking_status": data.smoking_status,
        "exercise_level": data.exercise_level
    }])

    prediction = diabetes_model.predict(input_df)[0]

    probability = diabetes_model.predict_proba(
        input_df
    )[0][1]

    risk_score = round(probability * 100)

    risk_level = (
        "Low Risk"
        if risk_score < 40
        else "Moderate Risk"
        if risk_score < 70
        else "High Risk"
    )

    return {
        "prediction": int(prediction),
        "risk_score": risk_score,
        "risk_level": risk_level
    }

@app.post("/predict/ckd")
def predict_ckd(data: CKDRequest):

    input_df = pd.DataFrame([{
        "age": data.age,
        "bmi": data.bmi,
        "systolic_bp": data.systolic_bp,
        "diastolic_bp": data.diastolic_bp,
        "creatinine": data.creatinine,
        "eGFR": data.eGFR,
        "sex": data.sex,
        "smoking_status": data.smoking_status,
        "charlson_index": data.charlson_index
    }])

    prediction = ckd_model.predict(input_df)[0]

    probability = ckd_model.predict_proba(input_df)[0][1]

    risk_score = round(probability * 100)

    risk_level = (
        "Low Risk"
        if risk_score < 40
        else "Moderate Risk"
        if risk_score < 70
        else "High Risk"
    )

    return {
        "prediction": int(prediction),
        "risk_score": risk_score,
        "risk_level": risk_level
    }    

@app.post("/predict/hyperlipidemia")
def predict_hyperlipidemia(
    data: HyperlipidemiaRequest
):

    input_df = pd.DataFrame([{
        "LDL": data.LDL,
        "HDL": data.HDL,
        "triglycerides": data.triglycerides,
        "total_cholesterol":
            data.total_cholesterol,
        "age": data.age,
        "bmi": data.bmi
    }])

    prediction = hyper_model.predict(input_df)[0]

    probability = hyper_model.predict_proba(
            input_df
        )[0][1]

    risk_score = round(probability * 100)

    risk_level = (
        "Low Risk"
        if risk_score < 40
        else "Moderate Risk"
        if risk_score < 70
        else "High Risk"
    )

    return {
        "prediction":
            int(prediction),
        "risk_score":
            risk_score,
        "risk_level":
            risk_level
    }

@app.post("/predict/cad")
def predict_cad(data: CADRequest):

    input_df = pd.DataFrame([{
        "age": data.age,
        "bmi": data.bmi,
        "systolic_bp": data.systolic_bp,
        "diastolic_bp": data.diastolic_bp,
        "LDL": data.LDL,
        "HDL": data.HDL,
        "triglycerides": data.triglycerides,
        "total_cholesterol": data.total_cholesterol,
        "BNP": data.BNP,
        "troponin_I": data.troponin_I,
        "sex": data.sex,
        "smoking_status": data.smoking_status,
        "charlson_index": data.charlson_index
    }])

    prediction = cad_model.predict(input_df)[0]

    probability = cad_model.predict_proba(input_df)[0][1]

    risk_score = round(probability * 100)

    risk_level = (
        "Low Risk"
        if risk_score < 40
        else "Moderate Risk"
        if risk_score < 70
        else "High Risk"
    )

    return {
        "prediction": int(prediction),
        "risk_score": risk_score,
        "risk_level": risk_level
    }

@app.post("/predict/heart-failure")
def predict_heart_failure(
    data: HeartFailureRequest
):

    input_df = pd.DataFrame([{
        "age": data.age,
        "bmi": data.bmi,
        "systolic_bp": data.systolic_bp,
        "diastolic_bp": data.diastolic_bp,
        "BNP": data.BNP,
        "troponin_I": data.troponin_I,
        "charlson_index": data.charlson_index,
        "sex": data.sex,
        "smoking_status": data.smoking_status
    }])

    prediction =heart_failure_model.predict(
            input_df
        )[0]

    probability =heart_failure_model.predict_proba(
            input_df
        )[0][1]

    risk_score =round(probability * 100)

    risk_level = (
        "Low Risk"
        if risk_score < 40
        else "Moderate Risk"
        if risk_score < 70
        else "High Risk"
    )

    return {
        "prediction": int(prediction),
        "risk_score": risk_score,
        "risk_level": risk_level
    }

