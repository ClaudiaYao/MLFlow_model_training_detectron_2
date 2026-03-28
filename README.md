# Introduction

The repo contains a sample model training project which will upload model training result to MLFlow server.  


## Steps to run the project:

1. Git clone.<br>
2. Install Python 3.11 or above version<br>
3. Create virtual environment: python -m venv .venv <br>
4. Initiate the virtual environment: .venv/Scripts/activate.ps1 (Windows PowerShell), .venv/bin/activate (Mac) <br>
5. Install dependencies: pip install -r requirements.txt (or use `uv sync`)
6. Create .env file under the project root path and fill in the correct info:
   ```
   MLFLOW_URI=
   MODEL_NAME=
   EXPERIMENT_NAME=
   ```
7. Recreate the src/data folder by referring to the post: [Fine-Tuning Detectron2 in Custom Object Recognition Tasks](https://medium.com/@claudia.yao2012/fine-tuning-detectron2-in-custom-object-recognition-tasks-235f4f914b7b)
8. If necessary, start MLflow server by referring to the post: [Setting Up an MLflow Tracking Server for Machine Learning Experiments (Part I)](https://medium.com/@claudia.yao2012/setting-up-an-mlflow-tracking-server-for-machine-learning-experiments-part-i-49d06262e67e)
6. Run the command: python src/main.py
