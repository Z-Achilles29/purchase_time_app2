# importing the required libraries
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
#using flask API

app = Flask(__name__)

model = joblib.load("purchase_time_xgb.pkl")
feature_order = joblib.load("feature_order.pkl")

def classify_intent(hours):
    if hours < 0.25:
        return "Very High Intent"
    elif hours < 0.75:
        return "High Intent"
    elif hours < 1.75:
        return "Moderate Intent"
    else:
        return "Low Intent"
    
def prediction_confidence(hours):
    """
    Confidence is derived from model stability.
    Mid-range predictions are usually more reliable.
    """
    if 0.25 <= hours <= 1.5:
        return "High"
    elif 0.15 <= hours <= 2.5:
        return "Medium"
    else:
        return "Low"



def recommended_action(intent):
    if intent == "Very High Intent":
        return "Avoid interruptions. Customer is likely to complete purchase naturally."
    elif intent == "High Intent":
        return "Provide subtle nudges like recommendations or free shipping reminders."
    elif intent == "Moderate Intent":
        return "Trigger reminder notifications or personalized suggestions."
    else:
        return "Consider promotional offers or discounts to encourage purchase."


def readable_time(hours):
    minutes = int(hours * 60)
    if minutes < 15:
        return "Within 15 minutes"
    elif minutes < 45:
        return "Within 30–45 minutes"
    elif minutes < 90:
        return "Within 1–1.5 hours"
    elif minutes < 180:
        return "Within 2–3 hours"
    else:
        return "More than 3 hours"
    
def uncertainty_minutes():
    return 15


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    message = None
    inputs = {
        "total_events": 0,
        "add_to_cart_events": 0,
        "page_view_events": 0,
        "total_amount_usd": 0
    }

    if request.method == "POST":
        inputs = {
            "total_events": int(request.form["total_events"]),
            "add_to_cart_events": int(request.form["add_to_cart_events"]),
            "page_view_events": int(request.form["page_view_events"]),
            "total_amount_usd": float(request.form["total_amount_usd"])
        }

    X = pd.DataFrame([inputs])[feature_order]
    y_log = model.predict(X)

    prediction = round(np.expm1(y_log[0]), 2)

    intent = classify_intent(prediction)
    action = recommended_action(intent)
    readable = readable_time(prediction)
    confidence = prediction_confidence(prediction)
    uncertainty = uncertainty_minutes()

    return render_template(
    "index.html",
    prediction=prediction,
    readable=readable,
    intent=intent,
    action=action,
    confidence=confidence,
    uncertainty=uncertainty,
    inputs=inputs
)



if __name__ == "__main__":
    app.run(debug=True)
