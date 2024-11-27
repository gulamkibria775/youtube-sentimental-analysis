import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

@pytest.mark.parametrize("model_name, stage", [
    ("yt_chrome_plugin_model", "staging"),
])
def test_load_latest_staging_model(model_name, stage):
    client = MlflowClient()

    # Fetch all registered models
    registered_models = client.search_model_versions(f"name='{model_name}'")

    # Find the latest model in the specified stage
    staging_models = [model for model in registered_models if model.current_stage.lower() == stage.lower()]
    latest_version = staging_models[0].version if staging_models else None

    # Validate existence of a staging model
    assert latest_version is not None, f"No model found in the '{stage}' stage for '{model_name}'"

    try:
        # Load the latest model
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Validate successful loading
        assert model is not None, "Failed to load the model"
        print(f"Model '{model_name}' version {latest_version} loaded successfully from '{stage}' stage.")
    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")

if __name__ == "__main__":
    test_model_name = "yt_chrome_plugin_model"
    test_stage = "staging"

    # Directly test the function
    client = MlflowClient()
    try:
        registered_models = client.search_model_versions(f"name='{test_model_name}'")
        staging_models = [model for model in registered_models if model.current_stage.lower() == test_stage.lower()]

        if staging_models:
            latest_version = staging_models[0].version
            model_uri = f"models:/{test_model_name}/{latest_version}"
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"Model '{test_model_name}' version {latest_version} loaded successfully from '{test_stage}' stage.")
        else:
            print(f"No model found in the '{test_stage}' stage for '{test_model_name}'.")
    except Exception as e:
        print(f"Error: {e}")
