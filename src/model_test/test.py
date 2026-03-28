import sys
from pathlib import Path

# The data and demo project is adpted from: https://www.kaggle.com/code/emineyetm/telco-customer-churn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_workflow.logging_register import log_register_model
from src.config import setting
from src.model_workflow.train import TrainPipeline
from src.model_workflow.dataset_info import generate_train_test_datasets
from src.model_test.inference import Detectron2InferencePipeline


def test_load_dataset():
    train_loader, test_loader, label_names = generate_train_test_datasets()
    assert train_loader is not None, "Train loader should not be None."
    assert test_loader is not None, "Test loader should not be None."
    assert label_names is not None, "Label names should not be None."
    print("✅ Dataset loading test passed.")

def test_model_train_and_register():
    train_loader, val_loader, label_names = generate_train_test_datasets()
    trainer = TrainPipeline()
    trained_model, parameters, metrics, signature, input_example, train_losses, val_losses = trainer.train(train_loader=train_loader,
                  val_loader=val_loader,
                    epochs=1)  # Use fewer epochs for testing
    print("✅ Model training test passed.")

    chart_saved_path = trainer.generate_loss_trend_chart(train_losses, val_losses)
    print("generated loss trend chart:", chart_saved_path)

    log_register_model(trained_model, setting.model_name, "detectron2", metrics, parameters, chart_saved_path, signature, input_example, register=False)
    print("✅ Model registration test passed.")

def test_model_inference():
    # need to train fiirst and then do the prediction test, as the inference pipeline depends on the trained model and the same preprocessing steps
    train_loader, val_loader, label_names = generate_train_test_datasets()
    trainer = TrainPipeline()
    trained_model, parameters, metrics, signature, input_example, train_losses, val_losses = trainer.train(train_loader=train_loader,
                  val_loader=val_loader,
                    epochs=1)  # Use fewer epochs for testing
    print("✅ Model training test passed.")

    inference_pipeline = Detectron2InferencePipeline(trained_model)
    # sample_input = next(iter(val_loader))

    sample_image_path = "C:\\Users\\claudia.yao\\ALL_MY_DOC\\MLFlow-model-training-detectron-2\\src\\data\\outdoor_garbage_dataset\\img\\5.jpg"
    predictions = inference_pipeline.predict(sample_image_path)
    assert predictions is not None, "Predictions should not be None."
    print("✅ Model inference test passed.")
    print("Predictions:", predictions)



if __name__ == "__main__":
    # test_load_dataset()
    # test_model_train_and_register()
    test_model_inference()