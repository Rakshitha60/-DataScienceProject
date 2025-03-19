from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Function to preprocess input data
def preprocess_input(data):
    # Convert the data into a DataFrame
    df = pd.DataFrame(data, index=[0])

    # Ensure all expected columns are present
    expected_columns = [
        'Age', 'SystolicBP', 'DiastolicBP', 'RBS', 'HB', 'TSH', 'Weight', 'Pregnacy_number',
        'HIV', 'HBsAg', 'PLSCS', 'PA',
        'Blood_Group_A+', 'Blood_Group_A-', 'Blood_Group_AB+',
        'Blood_Group_B+', 'Blood_Group_B-', 'Blood_Group_O+', 'Blood_Group_O-'
    ]

    # Ensure columns are in the right order and fill missing with zeros
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training data order
    df = df.reindex(columns=expected_columns, fill_value=0)

    return df


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from the form
    data = {
        'Age': float(request.form['Age']) if request.form['Age'] else 0.0,
        'SystolicBP': float(request.form['SystolicBP']) if request.form['SystolicBP'] else 0.0,
        'DiastolicBP': float(request.form['DiastolicBP']) if request.form['DiastolicBP'] else 0.0,
        'RBS': float(request.form['RBS']) if request.form['RBS'] else 0.0,
        'HB': float(request.form['HB']) if request.form['HB'] else 0.0,
        'TSH': float(request.form['TSH']) if request.form['TSH'] else 0.0,
        'Weight': float(request.form['Weight']) if request.form['Weight'] else 0.0,
        'Pregnacy_number': float(request.form['Pregnacy_number']) if request.form['Pregnacy_number'] else 0.0,
        'HIV_PS': 1 if request.form['HIV'] == 'PS' else 0,
        'HIV_NR': 1 if request.form['HIV'] == 'NR' else 0,
        'HBsAg_PS': 1 if request.form['HBsAg'] == 'PS' else 0,
        'HBsAg_NR': 1 if request.form['HBsAg'] == 'NR' else 0,
        'PLSCS_yes': 1 if request.form['PLSCS'] == 'yes' else 0,
        'PLSCS_no': 1 if request.form['PLSCS'] == 'no' else 0,
        'PA_yes': 1 if request.form['PA'] == 'yes' else 0,
        'PA_no': 1 if request.form['PA'] == 'no' else 0,
        'Blood_Group_A+': 1 if request.form['Blood_Group'] == 'A+' else 0,
        'Blood_Group_A-': 1 if request.form['Blood_Group'] == 'A-' else 0,
        'Blood_Group_AB+': 1 if request.form['Blood_Group'] == 'AB+' else 0,
        'Blood_Group_B+': 1 if request.form['Blood_Group'] == 'B+' else 0,
        'Blood_Group_B-': 1 if request.form['Blood_Group'] == 'B-' else 0,
        'Blood_Group_O+': 1 if request.form['Blood_Group'] == 'O+' else 0,
        'Blood_Group_O-': 1 if request.form['Blood_Group'] == 'O-' else 0,
    }

    # Preprocess the input data
    input_data = preprocess_input(data)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Return the result
    if prediction[0] == 1:
        prediction_text = 'High Risk'
    else:
        prediction_text = 'Low Risk'

    return render_template('index.html', prediction_text='Predicted Risk Level: {}'.format(prediction_text))


if __name__ == '__main__':
    app.run(debug=True)
