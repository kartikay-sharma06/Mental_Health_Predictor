"""Flask interface for the mental health score estimator."""

from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

from src.utils import load_object


BASE_DIR = Path(__file__).resolve().parent
FEATURES = [
    "Age",
    "Gender",
    "Country",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Study_Hours",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
    "Stress_Level",
]
CHOICES = {
    "Gender": ["Female", "Male"],
    "Academic_Level": ["High School", "Undergraduate", "Graduate"],
    "Most_Used_Platform": [
        "Facebook", "Instagram", "KakaoTalk", "LINE", "LinkedIn", "Snapchat",
        "TikTok", "Twitter", "VKontakte", "WeChat", "WhatsApp", "YouTube",
    ],
    "Purpose_Of_Use": ["Education", "Entertainment", "Networking", "News"],
    "Stress_Level": ["Low", "Medium", "High", "Very High"],
}


def load_countries():
    """Use the dataset's known country values so submitted categories match training."""
    csv_path = BASE_DIR / "artifacts" / "raw.csv"
    if csv_path.exists():
        return sorted(pd.read_csv(csv_path, usecols=["Country"])["Country"].dropna().astype(str).str.strip().unique())
    return ["Other"]


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
COUNTRIES = load_countries()
MODEL = load_object(str(BASE_DIR / "artifacts" / "model.pkl"))
PREPROCESSOR = load_object(str(BASE_DIR / "artifacts" / "preprocessor.pkl"))


@app.get("/")
def index():
    return render_template(
        "index.html", choices=CHOICES, countries=COUNTRIES,
        values={}, score=None, error=None,
    )


@app.post("/predict")
def predict():
    form = request.form
    values = {key: form.get(key, "").strip() for key in ["name", *FEATURES]}
    error = None
    score = None

    if not values["name"]:
        error = "Please enter your name."
    elif any(not values[key] for key in FEATURES):
        error = "Please complete every field before continuing."
    elif values["Country"] not in COUNTRIES:
        error = "Choose a country from the list."
    else:
        try:
            numeric = {
                "Age": int(values["Age"]),
                "Avg_Daily_Usage_Hours": float(values["Avg_Daily_Usage_Hours"]),
                "Daily_Unlocks": int(values["Daily_Unlocks"]),
                "Study_Hours": float(values["Study_Hours"]),
                "Physical_Activity_Hours": float(values["Physical_Activity_Hours"]),
                "Sleep_Hours_Per_Night": float(values["Sleep_Hours_Per_Night"]),
            }
            if not 13 <= numeric["Age"] <= 100:
                raise ValueError("Age must be between 13 and 100.")
            for key in ["Avg_Daily_Usage_Hours", "Study_Hours", "Physical_Activity_Hours", "Sleep_Hours_Per_Night"]:
                if not 0 <= numeric[key] <= 24:
                    raise ValueError("Hours must be between 0 and 24.")
            if numeric["Daily_Unlocks"] < 0:
                raise ValueError("Daily unlocks cannot be negative.")

            record = {key: values[key] for key in FEATURES}
            record.update(numeric)
            features = pd.DataFrame([record], columns=FEATURES)
            transformed = PREPROCESSOR.transform(features)
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            score = float(MODEL.predict(transformed)[0])
        except ValueError as exc:
            error = str(exc) or "Enter valid numbers in the numeric fields."
        except (TypeError, OverflowError):
            error = "Enter valid numbers in the numeric fields."
        except Exception:
            app.logger.exception("Prediction failed")
            error = "We couldn't calculate a score right now. Please try again."

    return render_template(
        "index.html", choices=CHOICES, countries=COUNTRIES,
        values=values, error=error, score=score,
    )


if __name__ == "__main__":
    app.run(debug=True)
