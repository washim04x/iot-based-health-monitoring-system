import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix , roc_auc_score , classification_report , accuracy_score , precision_recall_curve
import mlflow
import os
import logging
import json

#setup dagshub logging

dagshub_key=os.getenv("IOT_DAGSHUB_KEY")
if not dagshub_key:
    raise ValueError("DAGSHUB_KEY environment variable not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_key
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_key


mlflow.set_tracking_uri("https://dagshub.com/washim04x/iot-based-health-monitoring-system.mlflow")

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler =logging.FileHandler('model_evaluation.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)




def load_data(data_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(data_path)

def load_model(model_path):
    """Load a pre-trained model from a file."""
    return joblib.load(model_path)


def evaluate_model(model, transformer, data):
    """Evaluate the model and log metrics."""
    
    X_test = data.drop(columns=["HeartDisease"])
    X_test = transformer.transform(X_test)
    y_test = data["HeartDisease"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    

    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "classification_report": report,
        "precision": precision, 
        "recall": recall
    }

    logger.debug("mode evaluation metrics calculated successfully")

    return metrics

def save_metrics(metrics, path):
    """Save evaluation metrics to a file."""
    with open(path, "w") as f:
        json.dump(metrics, f)

def save_model_info(run_id, model_path, file_path) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    mlflow.set_experiment("Model_Evaluation_Experiment")
    with mlflow.start_run() as run:
        try:
            curr_dir = Path(__file__)
            home_dir = curr_dir.parent.parent.parent
            data_path = home_dir.as_posix() + "/data/processed/test.csv"
            model_path = home_dir.as_posix() + "/models/model.joblib"
            build_feature_path = home_dir.as_posix() + "/models/build_features_transformer.joblib"


            clf = load_model(model_path)
            data = load_data(data_path)
            features_transformer = load_model(build_feature_path)
            

            metrics = evaluate_model(clf, features_transformer, data)

            # Log only scalar numeric metrics to MLflow (filter out arrays and strings)
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float, np.integer, np.floating)):
                    mlflow.log_metric(metric_name, float(metric_value))
                else:
                    logger.debug(f"Skipping non-scalar metric '{metric_name}' for MLflow logging")

            evaluation_dir = home_dir.as_posix() + "/reports/model_evaluation_metrics.json"

            # Convert numpy arrays to lists for JSON serialization
            metrics_serializable = {}
            for k, v in metrics.items():
                if isinstance(v, np.ndarray):
                    metrics_serializable[k] = v.tolist()
                else:
                    metrics_serializable[k] = v
            
            save_metrics(metrics_serializable, evaluation_dir)

            # Log the model parameters to MLflow
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            save_model_info(run.info.run_id, model_path, "reports/model_info.json")

            # Log model to MLflow (matching working mlops-min-project pattern)
            mlflow.sklearn.log_model(clf, "model")
            # Log transformer as a separate model for easy loading
            mlflow.sklearn.log_model(features_transformer, "transformer")
            logger.debug("Model and transformer logged successfully to MLflow")

            # Log artifacts
            mlflow.log_artifact("reports/model_evaluation_metrics.json", artifact_path="model_evaluation")
            mlflow.log_artifact("reports/model_info.json", artifact_path="model_evaluation")
            mlflow.log_artifact("model_evaluation.log")
            
        except Exception as e:
            logger.error(f'Failed to complete the model evaluation process: {e}')
            print(f"Error: {e}")
            raise
        
        


if __name__ == "__main__":
    main()

