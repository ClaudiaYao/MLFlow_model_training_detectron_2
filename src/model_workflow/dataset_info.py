
import xml.etree.ElementTree as ET
import pandas as pd
from src.config import setting
from sklearn.model_selection import train_test_split
from src.config import setting
from torch.utils.data import DataLoader
from src.model_workflow.preprocess import transform
from src.model_workflow.preprocess import CSVImageDataset


def generate_dataset_df_from_annotation():
    tree = ET.parse(setting.data_dir / "annotations.xml")
    root = tree.getroot()


    label_names = [label.find("name").text for label in root.findall("meta/job/labels/label")]
    print(label_names)

    rows = []
    for img in root.findall(".//image"):
        name = img.attrib["name"]
        tags = [tag.attrib["label"] for tag in img.findall("tag")]

        if not tags:
            continue  # skip unlabeled images

        rows.append({
            "image_path": name,
            "is_empty": 1 if "is_empty" in tags else 0,
            "is_full": 1 if "is_full" in tags else 0,
            "is_scattered": 1 if "is_scattered" in tags else 0,
        })


    data_info = pd.DataFrame(rows)
    data_info["stratify_col"] = data_info[['is_empty', 'is_full', 'is_scattered']].astype(str).agg('-'.join, axis=1)
    data_info.head()
    data_info.to_csv(setting.data_dir / "data.csv", index=False)

    return data_info, label_names

def stratified_split_dataset(data_info_df: pd.DataFrame, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        data_info_df,
        test_size=test_size,
        stratify=data_info_df["stratify_col"],
        random_state=random_state
    )

    train_df = train_df.drop("stratify_col", axis = 1)
    train_df.to_csv(setting.data_dir / "train.csv", index=False)

    test_df = test_df.drop("stratify_col", axis = 1)
    test_df.to_csv(setting.data_dir / "test.csv", index=False)

    return train_df, test_df


def generate_train_test_datasets():
    data_info_df, label_names = generate_dataset_df_from_annotation()
    train_df, test_df = stratified_split_dataset(data_info_df)

    image_paths = [setting.data_dir / "img" / img_name for img_name in train_df["image_path"]]
    labels = train_df[label_names].values
    train_dataset = CSVImageDataset(
        image_paths=image_paths,
        labels=labels,
        transform=transform
    )

    test_image_paths = [setting.data_dir / "img" / img_name for img_name in test_df["image_path"]]
    test_labels = test_df[label_names].values
    test_dataset = CSVImageDataset(
        image_paths=test_image_paths,
        labels=test_labels,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

    return train_loader, test_loader, label_names

