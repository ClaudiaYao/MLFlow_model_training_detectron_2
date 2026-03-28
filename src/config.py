import os, sys
from dotenv import load_dotenv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

class Config:
    mlflow_uri = os.getenv("MLFLOW_URI")
    model_name = os.getenv("MODEL_NAME")
    artifact_name = os.getenv("ARTIFACT_NAME", model_name)
    experiment_name = os.getenv("EXPERIMENT_NAME")
    data_dir = PROJECT_ROOT / "src" / "data" / "outdoor_garbage_dataset"

setting = Config()