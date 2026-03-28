
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

# Transforms (resize + crop + normalize for DINOv2)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),   # DINOv2 expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])


class CSVImageDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        # self.df = pd.read_csv(csv_file)
        self.image_paths = image_paths
        self.labels = labels
        self.preprocess = transform

        # List of label columns in the CSV
        self.label_columns = ["is_empty", "is_full", "is_scattered"]
        # self.label_columns = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.preprocess:
            image = self.preprocess(image)

        # Build multi-hot label vector
        label = torch.tensor([float(val) for val in self.labels[idx]], dtype=torch.float32)

        return image, label
