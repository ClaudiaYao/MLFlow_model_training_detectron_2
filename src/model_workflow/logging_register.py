import mlflow
from mlflow.exceptions import RestException
import torch
from src.config import setting
from src import config 

        
def log_register_model(
    model,
    model_name,
    model_type,
    metrics: dict,
    parameters: dict,
    artifact_chart_path,
    signature,
    input_examples,
    input_examples_gt_output,
    label_names,
    register=True
):
    """Log model .pth artifact to S3 via MLflow"""

    mlflow.set_tracking_uri(setting.mlflow_uri)
    client = mlflow.tracking.MlflowClient()

    print("🔄 Logging model artifact to S3...")

    mlflow.set_experiment(setting.experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        parameters["run_id"] = run_id

        # Metrics & params
        mlflow.log_metrics(metrics)
        mlflow.log_params(parameters)
        # mlflow.log_inputs(input_example)
    
        mlflow.set_tag("author", "Claudia")
        mlflow.set_tag("model_type", model_type)

        print(f"Artifact URI: {mlflow.get_artifact_uri()}")

        # Log evaluation chart if exists
        if artifact_chart_path:
            print("Logging evaluation chart artifacts...", artifact_chart_path)
            mlflow.log_artifact(artifact_chart_path)

        # -------------------------------
        # 🔑 Save model .pth
        # -------------------------------
        local_model_path = config.PROJECT_ROOT / f"{model_name}.pth"
        torch.save(model.state_dict(), local_model_path)
        print(f"Saving model to {local_model_path}")

        # Log to MLflow artifact store (S3)
        mlflow.log_artifact(local_model_path, artifact_path="model")
        print("✅ Model artifact logged to S3.")

        # Log model signature and input example
        mlflow.log_dict(signature.to_dict(), "model_signature.json")
        mlflow.log_dict({"input_example": input_examples}, "input_example.json")
        mlflow.log_dict({"input_example_gt_output": input_examples_gt_output}, "input_example_gt_output.json")
        mlflow.log_dict({"label_names": label_names}, "label_names.json")

        # -------------------------------
        # Optional: Registry model version
        # -------------------------------
        if register:

            try:
                client.get_registered_model(model_name)
            except RestException:
                client.create_registered_model(model_name)

            model_uri = f"runs:/{run_id}/model"
            mv = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id
            )

            print(f"✅ Registered model version: {mv.version}")

            client.set_registered_model_alias(
                name=model_name,
                alias="latest-model",
                version=mv.version,
            )

            return mv.version
        return None
