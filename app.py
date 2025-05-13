from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


with open('placement.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            cgpa = float(request.form['cgpa'])
            iq = float(request.form['iq'])

            
            input_features = scaler.transform([[cgpa, iq]])
            result = model.predict(input_features)

            print(f"Model prediction: {result}")

            predicted_value = int(result[0])
            prediction = "Placed" if predicted_value == 1 else "Not Placed"

        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
