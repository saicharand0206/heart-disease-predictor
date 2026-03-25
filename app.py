import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model

# Load your model
model = load_model("heart.h5")

# Prediction function
def predict(age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal):

    data = np.array([[age, sex, cp, trestbps, chol, fbs,
                      restecg, thalach, exang, oldpeak,
                      slope, ca, thal]])

    prediction = model.predict(data)

    if prediction[0][0] > 0.5:
        return "High Risk of Heart Disease"
    else:
        return "Low Risk"

# Interface
app = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Sex (0 Female, 1 Male)"),
        gr.Number(label="Chest Pain (0-3)"),
        gr.Number(label="Resting BP"),
        gr.Number(label="Cholesterol"),
        gr.Number(label="FBS >120"),
        gr.Number(label="Rest ECG (0-2)"),
        gr.Number(label="Max Heart Rate"),
        gr.Number(label="Exercise Angina"),
        gr.Number(label="Oldpeak"),
        gr.Number(label="Slope (0-2)"),
        gr.Number(label="CA (0-3)"),
        gr.Number(label="Thal (0-3)")
    ],
    outputs="text"
)

app.launch()