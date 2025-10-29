import os
import mlflow

def promote_model():

    dagshub_token = os.getenv("iot-dagshub-key")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_KEY environment variable is not set")
    
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "washim04x"
    repo_name = "iot-based-health-monitoring-system"
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    client = mlflow.MlflowClient()

    # List of models to promote
    models_to_promote = ["model", "transformer"]
    
    for model_name in models_to_promote:
        try:
            # Get the latest version in staging
            latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

            # Archive the current production model
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            for version in prod_versions:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )

            # Promote the new model to production
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version_staging,
                stage="Production"
            )
            print(f"{model_name} version {latest_version_staging} promoted to Production")
        except Exception as e:
            print(f"Error promoting {model_name}: {e}")

if __name__ == "__main__":
    promote_model()
