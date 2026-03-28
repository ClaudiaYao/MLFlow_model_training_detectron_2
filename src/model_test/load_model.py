import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from config import setting

def load_model_safe(model_name, version_id=None, alias=None):
    """
    Load model from MLflow registry or runs.
    
    Args:
        model_name: Name of the registered model
        version_id: Specific version number to load
        alias: Model alias to load (e.g., "latest-model", "champion")
    
    Returns:
        Loaded MLflow model or None if loading fails
    """
    mlflow.set_tracking_uri(setting.mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Option 1: Load by alias (recommended for MLFlow 3.8)
        if alias:
            model_uri = f"models:/{model_name}@{alias}"
            print(f"Loading model from alias: {model_uri}")
            return mlflow.pyfunc.load_model(model_uri)
        
        # Option 2: Load by specific version
        elif version_id:
            model_uri = f"models:/{model_name}/{version_id}"
            print(f"Loading model from version: {model_uri}")
            return mlflow.pyfunc.load_model(model_uri)
        
        # Option 3: Load latest version from registry
        else:
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                print(f"No versions found for model '{model_name}'")
                return None
            
            # Get the latest version
            latest_version = max(versions, key=lambda v: int(v.version))
            model_uri = f"models:/{model_name}/{latest_version.version}"
            print(f"Loading latest model version: {model_uri}")
            return mlflow.pyfunc.load_model(model_uri)
            
    except mlflow.exceptions.MlflowException as e:
        print(f"MLflow error loading model '{model_name}': {e}")
        
        # Fallback: Load from latest run if registry fails
        print("Attempting to load from latest run...")
        try:
            experiment = client.get_experiment_by_name(setting.experiment_name)
            if not experiment:
                print(f"Experiment '{setting.experiment_name}' not found")
                return None
                
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id], 
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if not runs:
                print("No runs found in experiment")
                return None
                
            latest_run = runs[0]
            run_id = latest_run.info.run_id
            
            # Use the correct artifact path from logging_register.py
            model_uri = f"runs:/{run_id}/model"  # or "local_stored" if register=False
            print(f"Loading model from run: {model_uri}")
            return mlflow.pyfunc.load_model(model_uri)
            
        except Exception as fallback_error:
            print(f"Fallback failed: {fallback_error}")
            return None
    
    except Exception as e:
        print(f"Failed to load model '{model_name}': {e}")
        return None
