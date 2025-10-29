import mlflow
import json
from pathlib import Path
import logging
import os

dagshub_key=os.getenv("iot-dagshub-key")
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




def load_model_info(file_path) :
    """Load the model run ID and path from a JSON file."""
    try :
        with open(file_path , 'r') as file :
            model_info= json.load(file)
            logger.debug('Model info loaded from %s' , file_path)
            return model_info
    except Exception as e :
        logger.error('Error occurred while loading the model info: %s' , e)
        raise

def registered_model(model_info, model_name, artifact_path):
    """Register the model in MLflow Model Registry."""
    run_id = model_info['run_id']

    logger.debug('Registering %s with run ID: %s and artifact path: %s', model_name, run_id, artifact_path)

    try:
        model_version = mlflow.register_model(
            model_uri=f"runs:/{run_id}/{artifact_path}",
            name=model_name
        )
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
        )
        logger.info('%s version %s registered and promoted to Staging', model_name, model_version.version)
    except Exception as e:
        logger.error('Error occurred while registering %s: %s', model_name, e)
        raise

if __name__ == "__main__":
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    model_info_path = home_dir.as_posix() + "/reports/model_info.json"
    model_info = load_model_info(model_info_path)
    
    # Register and transition the main model to Staging
    registered_model(model_info, "model", "model")
    
    # Register and transition the transformer to Staging
    registered_model(model_info, "transformer", "transformer") 