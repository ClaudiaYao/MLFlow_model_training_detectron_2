
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
from src import config 
from sklearn.metrics import f1_score, precision_score, recall_score
from src.model_workflow.preprocess import transform  # Assuming preprocessing logic is here

import torch
import torch.nn as nn
import torch.optim as optim
from src.model_workflow.model_def import DinoV2Classifier
import io
import base64
import numpy as np


class TrainPipeline:
    def __init__(self, device="cpu"):
        """
        Initialize the training pipeline.
        :param model: The model to train.
        :param optimizer: Optimizer for training.
        :param loss_fn: Loss function.
        :param device: Device to use ('cpu' or 'cuda').
        """
        self.THRESHOLD = 0.5

        self.model  = DinoV2Classifier(num_classes=3)
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        self.loss_fn = nn.BCEWithLogitsLoss()  # multi-label
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.transform = transform 


   
    def train(self, train_loader, val_loader=None, epochs=10):
        """
        Train the model.
        :param data_loader: DataLoader providing batches of (image, label).
        :param epochs: Number of training epochs.
        """
        train_losses = []
        val_losses = []

        for epoch in range(epochs):  # adjust epochs
            self.model.train()
            train_loss = 0
            all_labels = []
            all_preds = []

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels.float())

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                # Apply sigmoid + threshold to get predictions
                preds = (torch.sigmoid(outputs) > self.THRESHOLD).float()

                all_labels.append(labels.cpu())
                all_preds.append(preds.cpu())

            # Stack all batches
            all_labels = torch.cat(all_labels, dim=0).numpy()
            all_preds = torch.cat(all_preds, dim=0).numpy()

            # Calculate multi-label metrics
            micro_f1 = f1_score(all_labels, all_preds, average="micro")
            macro_f1 = f1_score(all_labels, all_preds, average="macro")

            avg_loss = train_loss / len(train_loader)
            train_losses.append(avg_loss)
            precision = precision_score(all_labels, all_preds, average="micro")
            recall = recall_score(all_labels, all_preds, average="micro")

            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

            metrics = {
            "train_micro_f1": micro_f1,
            "train_macro_f1": macro_f1,
            "train_avg_loss": avg_loss,
            "train_precision": precision,
            "train_recall": recall,
            }

            if val_loader:
                val_loss, val_micro_f1, val_macro_f1, val_precision, val_recall = self.evaluate(val_loader)
                metrics.update({
                    "val_micro_f1": val_micro_f1,
                    "val_macro_f1": val_macro_f1,
                    "val_avg_loss": val_loss,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                })
                val_losses.append(val_loss)

        # record parameters, metrics, and model signature for MLflow
        images, labels = next(iter(train_loader))  # Get a batch of data
        input_example = images[:1]  # Use the first image as an example
        input_example_np = input_example.cpu().numpy()  # Convert to NumPy array
        input_example_gt_output = labels[:1]  # Corresponding label for the input example
        signature = infer_signature(input_example_np, input_example_gt_output.cpu().numpy())
        
        # Create parameters dictionary
        parameters = {
            "n_train": len(train_loader.dataset),
            "n_val": len(val_loader.dataset) if val_loader else 0,
            "num_classes": labels.shape[1],  # Number of labels for multi-label classification
            "model_type": "Detectron2",
            "backbone": "DINOv2",
            "threshold": self.THRESHOLD,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
        }

        return self.model, parameters, metrics, signature, input_example, input_example_gt_output, train_losses, val_losses



    def evaluate(self, test_loader):
        """
        Evaluate the model on the test set.
        :param test_loader: DataLoader for test data.
        """
        self.model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():  # Disable gradient computation for evaluation
            for images, labels in test_loader:
                # Move data to the appropriate device
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                test_loss += loss.item()

                # Apply sigmoid + threshold to get predictions
                preds = (torch.sigmoid(outputs) > self.THRESHOLD).float()

                all_labels.append(labels.cpu())
                all_preds.append(preds.cpu())

            # Stack all batches
            all_labels = torch.cat(all_labels, dim=0).numpy()
            all_preds = torch.cat(all_preds, dim=0).numpy()

            micro_f1 = f1_score(all_labels, all_preds, average="micro")
            macro_f1 = f1_score(all_labels, all_preds, average="macro")

            precision = precision_score(all_labels, all_preds, average="micro")
            recall = recall_score(all_labels, all_preds, average="micro")

            avg_loss = test_loss / len(test_loader)

        print(f"Test Loss: {avg_loss:.4f}, Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        return avg_loss, micro_f1, macro_f1, precision, recall
  

    def generate_loss_trend_chart(self, train_losses: list, val_losses: list):
        """Generate evaluation chart data from evals_result."""
        try: 
            epochs = len(train_losses)
            x_axis = range(0, epochs)

            print("train loss:", train_losses)
            print("validation loss:", val_losses)
            
            # Plot Average Loss for each epoch
            plt.figure(figsize=(8, 5))
            plt.plot(x_axis, train_losses, label='Train Loss')
            plt.plot(x_axis, val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Log Loss')
            plt.title('Training and Validation Log Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            saved_path = config.PROJECT_ROOT / "loss_curve.png"
            plt.savefig(saved_path)  # Save the plot
            plt.close()

            return saved_path
        except Exception as e:
            print(f"Error generating evaluation chart: {e}")
            return None