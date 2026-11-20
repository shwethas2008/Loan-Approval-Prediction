import pandas as pd
import joblib

model = joblib.load("models/loan_model.pkl")

data = pd.DataFrame([{
    "annual_income": 650000,
    "credit_score": 750,
    "loan_amount": 200000,
    "employment_years": 5,
    "existing_debt": 30000
}])

prediction = model.predict(data)
print("Loan Status:", prediction[0])
 