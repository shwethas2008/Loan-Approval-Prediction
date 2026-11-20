from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Loan Approval Prediction")

model = joblib.load("models/loan_model.pkl")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


class LoanInput(BaseModel):
    annual_income: float
    credit_score: int
    loan_amount: float
    employment_years: int
    existing_debt: float


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/predict")
def predict(data: LoanInput):
    df = pd.DataFrame([data.model_dump()])
    result = model.predict(df)[0]
    return {"loan_status": str(result)}


@app.post("/submit")
def submit(
    request: Request,
    annual_income: float = Form(...),
    credit_score: int = Form(...),
    loan_amount: float = Form(...),
    employment_years: int = Form(...),
    existing_debt: float = Form(...)
):
    df = pd.DataFrame([{
        "annual_income": annual_income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "employment_years": employment_years,
        "existing_debt": existing_debt
    }])

    result = model.predict(df)[0]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"prediction": str(result)}
    )