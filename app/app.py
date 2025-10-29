from flask import Flask,render_template,request
import mlflow
import os


dagshub_key = os.getenv("IOT_DAGSHUB_KEY")
if not dagshub_key:
    raise ValueError("DAGSHUB_KEY environment variable not set")
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_key
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_key

mlflow.set_tracking_uri("https://dagshub.com/washim04x/iot-based-health-monitoring-system.mlflow")

def get_latest_model_version(model_name):
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(name=model_name, stages=["Production"])
    if not versions:
        raise ValueError(f"No versions found for model '{model_name}' in Production stage.")
    latest_version = max(versions, key=lambda v: v.version)
    return latest_version.version

#load model
model_name = "model"
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

#load transformer
transformer_name = "transformer"
transformer_version = get_latest_model_version(transformer_name)
transformer_uri = f"models:/{transformer_name}/{transformer_version}"
transformer = mlflow.sklearn.load_model(transformer_uri)


app = Flask(__name__)

@app.route('/')
def home():
    return  render_template('index.html',prediction=None)

@app.route('/predict',methods=['POST'])
def predict():
    try:
        # Get form data - numerical features
        age = float(request.form.get('Age'))
        resting_bp = float(request.form.get('RestingBP'))
        cholesterol = float(request.form.get('Cholesterol'))
        max_hr = float(request.form.get('MaxHR'))
        oldpeak = float(request.form.get('Oldpeak'))
        
        # Get form data - categorical features (as strings, matching training data)
        sex = request.form.get('Sex')  # 'M' or 'F'
        chest_pain_type = request.form.get('ChestPainType')  # 'ATA', 'NAP', 'ASY', 'TA'
        fasting_bs = int(request.form.get('FastingBS'))  # 0 or 1 (already numeric)
        resting_ecg = request.form.get('RestingECG')  # 'Normal', 'ST', 'LVH'
        exercise_angina = request.form.get('ExerciseAngina')  # 'N' or 'Y'
        st_slope = request.form.get('ST_Slope')  # 'Up', 'Flat', 'Down'
        
        # Create DataFrame with proper column names and data types (matching training data exactly)
        import pandas as pd
        input_data = {
            'Age': [age],
            'Sex': [sex],
            'ChestPainType': [chest_pain_type],
            'RestingBP': [resting_bp],
            'Cholesterol': [cholesterol],
            'FastingBS': [fasting_bs],
            'RestingECG': [resting_ecg],
            'MaxHR': [max_hr],
            'ExerciseAngina': [exercise_angina],
            'Oldpeak': [oldpeak],
            'ST_Slope': [st_slope]
        }
        input_df = pd.DataFrame(input_data)
        print("Input DataFrame:")
        print(input_df)
        print("Data types:", input_df.dtypes)
        
        # Transform features using the transformer
        features_transformed = transformer.transform(input_df)

        print("Transformed Features:")
        print(features_transformed)

        # Predict heart disease
        result = model.predict(features_transformed)
        
        print("Prediction Result:")
        print(result)
        print(result[0])

        # Return result (0 = No disease, 1 = Disease)
        return render_template('index.html', prediction=int(result[0]))
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', prediction=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)



