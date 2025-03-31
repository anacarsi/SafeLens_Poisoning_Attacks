# Class that handles loading the images without requiring them to be in subdirectories (ImageNet dataset)
from torch.utils.data import Dataset
from PIL import Image
import os


class CustomDataset(Dataset):
    def __init__(self, image_folder, val_labels=None, transform=None):
        """
        Custom Dataset for Tiny-ImageNet Validation Set.

        Args:
            image_folder (str): Path to the validation images folder.
            val_labels (dict, optional): Mapping from image filename to class index.
            transform (callable, optional): Transformations to apply to images.
        """
        self.image_folder = image_folder
        self.transform = transform
        self.val_labels = val_labels  # Store the label mapping

        # Collect all valid image paths
        self.image_paths = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.endswith(".JPEG")
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Extract filename from path
        image_name = os.path.basename(img_path)

        # Retrieve corresponding label from val_labels
        label = self.val_labels.get(image_name, -1)  # Default to -1 if not found

        if self.transform:
            image = self.transform(image)

        return image, label


def parse_val_annotations(annotation_file) -> dict:
    """
    Parse the ImageNet validation annotations file to extract the WordNet IDs.
    """
    val_labels = {}
    with open(annotation_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            image_name, wnid = parts[0], parts[1]  # Extract image filename and class
            val_labels[image_name] = wnid  # Store mapping (filename -> class label)
    return val_labels
