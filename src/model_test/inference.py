import mlflow
import torch
from torch.utils.data import DataLoader
from src.model_workflow.preprocess import transform
from PIL import Image
from src.model_workflow.preprocess import CSVImageDataset
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unnormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    return img


class Detectron2InferencePipeline:

    def __init__(self, model, device=None):

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Build model architecture
        self.model = model
        self.model.to(self.device)
        self.model.eval()

        # Preprocess & config
        self.preprocess = transform
        self.label_columns = ["is_empty", "is_full", "is_scattered"]
        self.idx_to_class = {
            0: "is_empty",
            1: "is_full",
            2: "is_scattered",
        }

        self.THRESHOLD = 0.5
        self.batch_size = 8

    
    def predict_single(self, image_full_path_name):        
        
        # image_full_name = "/content/drive/MyDrive/outdoor_garbage_dataset/img/51.jpg"
        image_open = Image.open(image_full_path_name)
        test_image = self.preprocess(image_open)
        img_tensor = test_image.unsqueeze(0)  # shape: [1, C, W, H]

        # Move to device
        img_tensor = img_tensor.to(device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)  # raw logits
            pred = (torch.sigmoid(output) > self.THRESHOLD).float()
            probs = torch.sigmoid(output)

        # Get predicted labels and true labels
        pred_labels = [self.idx_to_class[j] for j in range(len(self.label_columns)) if pred[0][j] == 1]

        print("pred_labels:", pred_labels)
        print("prob:", probs)

        pred_confidences = [probs[0][j].item() for j in range(len(self.label_columns)) if pred[0, j] == 1]

        return pred_labels, pred_confidences


    def predict_batch(self, image_full_path_names: list[str]):
        """
        Predict for a batch of images. Handles both directories and lists of file paths.
        :param image_full_path_names: List of image file paths or a directory containing images.
        :return: List of predictions.
        """

        # Otherwise, use a DataLoader for batch predictions
        dataset = CSVImageDataset(
            image_paths=image_full_path_names,
            labels=None,  # Labels are not needed for inference
            transform=self.preprocess
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []
        pred_confidences = []

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                outputs = self.model(batch)

                preds = (torch.sigmoid(outputs) > self.THRESHOLD).float()
                probs = torch.sigmoid(outputs)
                pred_confidences.extend(probs.cpu().numpy().tolist())
                for pred in preds:
                    pred_labels = [self.idx_to_class[j] for j in range(len(self.label_columns)) if pred[j] == 1]
                    predictions.append(pred_labels)

        return predictions, pred_confidences
    
    def predict(self, model_input):
    # Check if the input is a directory
        image_full_path_names = []
        if isinstance(model_input, str) and os.path.isdir(model_input):
            # Get all image files in the directory
            image_full_path_names = [
                os.path.join(model_input, f)
                for f in os.listdir(model_input)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
        else:
            image_full_path_names.append(model_input)

        # If there are fewer images than the batch size, use a loop for single predictions
        if len(image_full_path_names) < self.batch_size:
            prediction_result = [self.predict_single(image_path) for image_path in image_full_path_names]
            return prediction_result
        else:
            prediction_result = self.predict_batch(image_full_path_names)