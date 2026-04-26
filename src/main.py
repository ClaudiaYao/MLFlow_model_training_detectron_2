
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_workflow.logging_register import log_register_model
from src.config import setting
from src.model_workflow.train import TrainPipeline
from src.model_workflow.dataset_info import generate_train_test_datasets
from src.model_test.inference import Detectron2InferencePipeline

# test the whole training, logging, registering and inference workflow end to end. Note that the test is not designed for unit testing but more for integration testing, as the training and inference pipeline are closely connected and depend on each other. The test will train a model, log and register it, and then do the inference with the same trained model.

train_loader, val_loader, label_names = generate_train_test_datasets()
trainer = TrainPipeline()
trained_model, parameters, metrics, signature, input_example, train_losses, val_losses = trainer.train(train_loader=train_loader,
                val_loader=val_loader,
                epochs=8)  # Use fewer epochs for testing
print("✅ Model training test passed.")

chart_saved_path = trainer.generate_loss_trend_chart(train_losses, val_losses)
print("generated loss trend chart:", chart_saved_path)
print("✅ Loss trend chart generation test passed.")

log_register_model(trained_model, setting.model_name, "detectron2", metrics, parameters, chart_saved_path, signature, input_example, register=True)
print("✅ Model registration test passed.")

inference_pipeline = Detectron2InferencePipeline(trained_model)
sample_image_path = setting.data_dir / "img" / "5.jpg"
predictions = inference_pipeline.predict(sample_image_path)
assert predictions is not None, "Predictions should not be None."
print("✅ Model inference test passed.")
print("Predictions:", predictions)



