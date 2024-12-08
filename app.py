from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# Load your model (make sure it's available)
model = joblib.load('heart_disease_model.pkl')  # Replace with your model file path

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Sample values (these will be pre-filled in the form)
    sample_data = {
        "age": 50,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0
    }

    if request.method == 'POST':
        # Get form data from user input or use sample data
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])

        # Create an array of features
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang]])
        
        # Standardize features (if needed)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Send results to the HTML page
        return render_template('index.html', prediction=prediction[0], probability=probability,
                               form_data=request.form, sample_data=sample_data)
    
    # On initial load, pass sample data
    return render_template('index.html', prediction=None, probability=None, sample_data=sample_data)

if __name__ == '__main__':
    app.run(debug=True)
