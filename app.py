from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('placement_model.pkl')  # Load your trained model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input values from form
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    # Make prediction using model
    prediction = model.predict(final_features)

    # Prepare response
    output = prediction[0]

    return render_template('index.html', prediction_text=f'The student is {output}')

if __name__ == '__main__':
    app.run(debug=True)
