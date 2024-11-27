import mlflow
from mlflow.tracking import MlflowClient

def promote_model():
    # Set up the MLflow tracking URI (local or remote server)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Replace with your actual tracking URI if needed
    
    client = MlflowClient()

    model_name = "yt_chrome_plugin_model"

    # Get versions from staging
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])

    # Check if there is a model in the "Staging" stage
    if not staging_versions:
        print(f"No models found in Staging for model: {model_name}. Cannot promote.")
        return  # Exit the function if no staging model is found

    # If there are staging versions, get the latest one
    latest_version_staging = staging_versions[0].version
    print(f"Found latest staging version: {latest_version_staging}")

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
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()
