import sys
import os
import pandas as pd
from flask import Flask, request, render_template
import cloudpickle


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = Flask(__name__)

# Load both preprocessor pipeline and model
with open("models/preprocessor_v1.pkl", "rb") as f:
    preprocessor = cloudpickle.load(f)

with open("models/rf_model_v1.pkl", "rb") as f:
    model = cloudpickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form input
    user_input = {
        "longitude": float(request.form["longitude"]),
        "latitude": float(request.form["latitude"]),
        "housing_median_age": float(request.form["housing_median_age"]),
        "total_rooms": float(request.form["total_rooms"]),
        "total_bedrooms": float(request.form["total_bedrooms"]),
        "population": float(request.form["population"]),
        "households": float(request.form["households"]),
        "median_income": float(request.form["median_income"]),
        "ocean_proximity": request.form["ocean_proximity"]
    }

    input_df = pd.DataFrame([user_input])

    # Apply preprocessing pipeline first
    prepared_input = preprocessor.transform(input_df)

    # Then pass to trained model
    prediction = model.predict(prepared_input)[0]

    return render_template("result.html", prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
