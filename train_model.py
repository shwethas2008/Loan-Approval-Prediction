import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("data/loan_data.csv")

# 2. Check missing values
print("Missing Values:")
print(df.isnull().sum())

# 3. Remove missing target rows
df = df.dropna(subset=["loan_status"])

# 4. Separate input and output
X = df.drop(columns=["loan_status"])
y = df["loan_status"]

# 5. Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)

# 6. Missing value filling and model creation
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ))
])

# 7. MLflow setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Loan Approval Prediction")

with mlflow.start_run():

    # 8. Train model
    model.fit(X_train, y_train)

    # 9. Test model
    predictions = model.predict(X_test)

    # 10. Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", accuracy * 100, "%")
    print(classification_report(y_test, predictions))

    # 11. Save model
    joblib.dump(model, "models/loan_model.pkl")

    # 12. MLflow tracking
    mlflow.log_param("algorithm", "Decision Tree")
    mlflow.log_param("max_depth", 4)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_artifact("models/loan_model.pkl")

print("Model trained and saved successfully!")