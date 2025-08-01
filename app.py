from flask import Flask, request, render_template
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)

app = application

# route for home page
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoints():
    if request.method == 'POST':
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        prediction_pipeline = PredictionPipeline()
        result = prediction_pipeline.predict(pred_df)

        return render_template("home.html",results=result[0])

    return render_template("home.html")

if __name__ == "__main__":
    app.run('0.0.0.0',port=8000,debug=True)