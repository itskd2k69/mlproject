from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import logging

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize Flask app
application = Flask(__name__, template_folder="templates", static_folder="static")
app = application

# Configure logging
logging.basicConfig(level=logging.INFO)

# ----------------------- ROUTES -----------------------

@app.route("/")
def index():
    """Render the index (landing) page."""
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    """Handle data input and prediction logic."""
    try:
        if request.method == "GET":
            return render_template("home.html")

        # Extract data from form
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score")),
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        logging.info(f"Input DataFrame: \n{pred_df}")

        # Predict
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        logging.info(f"Prediction result: {results[0]}")

        return render_template("home.html", results=results[0])

    except Exception as e:
        logging.error("Error occurred during prediction", exc_info=True)
        return render_template("home.html", results="Error during prediction. Please try again.")

# ----------------------- MAIN -----------------------

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000)
