from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the models
rf_model = joblib.load('models/rf_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')


def predict_with_explanation(composition_data):
    """
    Predict crack susceptibility and provide explanation for new composition.
    """
    # Convert input into DataFrame
    new_composition = pd.DataFrame([composition_data])

    # Scale the input composition
    scaled_comp = scaler.transform(new_composition)

    # Make prediction
    pred_encoded = rf_model.predict(scaled_comp)
    pred_label = label_encoder.inverse_transform(pred_encoded)
    prob_scores = rf_model.predict_proba(scaled_comp)

    # Generate explanation
    explanation = {
        "prediction": pred_label[0],
        "probabilities": {
            label: float(prob_scores[0][i])
            for i, label in enumerate(label_encoder.classes_)
        },
        "top_elements": [
            {"element": element, "value": value}
            for element, value in composition_data.items()
        ]
    }

    return explanation


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form.to_dict()

        # Only take 5 selected elements
        composition = {
            'Al': float(data.get('Al', 0)),
            'Zn': float(data.get('Zn', 0)),
            'Cu': float(data.get('Cu', 0)),
            'Mg': float(data.get('Mg', 0)),
            'Si': float(data.get('Si', 0))
        }

        # Get prediction
        result = predict_with_explanation(composition)
        return render_template('result.html', result=result)

    except Exception as e:
        return f"An error occurred: {str(e)}", 400


if __name__ == '__main__':
    app.run(debug=True)
