import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import warnings
from typing import Tuple

# Albumentations transforms
basic_transform = A.Compose([
    A.Resize(64, 64),
    A.RandomCrop(64, 64),
    ToTensorV2()
])
advanced_transform = A.Compose([
    A.Resize(64, 64),
    A.RandomCrop(64, 64),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.RandomRotate90(),
    A.HueSaturationValue(),
    ToTensorV2()
])

# Albumentations wrapper
class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = np.array(img)  # Convert from PIL image to NumPy array
        augmented = self.transform(image=img)
        return augmented["image"]

# Filtered ImageFolder to handle invalid files
class FilteredImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        valid_samples = []
        for path, class_idx in self.samples:
            try:
                # Try opening the image to check for validity
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # Suppress warnings
                    img = Image.open(path)
                    img.verify()  # Verify image integrity
                valid_samples.append((path, class_idx))
            except (IOError, SyntaxError, ValueError) as e:
                print(f"Invalid image file {path}: {e}")
        self.samples = valid_samples
        self.targets = [s[1] for s in self.samples]

# Combined dataset for basic and advanced transforms
class CombinedDataset(Dataset):
    def __init__(self, dataset_path, basic_transform, advanced_transform):
        self.image_folder = FilteredImageFolder(root=dataset_path)
        self.basic_transform = AlbumentationsTransform(basic_transform)
        self.advanced_transform = AlbumentationsTransform(advanced_transform)

    def __len__(self):
        return 2 * len(self.image_folder)

    def __getitem__(self, idx):
        original_idx = idx % len(self.image_folder)
        img, label = self.image_folder[original_idx]

        if idx < len(self.image_folder):
            img = self.basic_transform(img)
        else:
            img = self.advanced_transform(img)

        return img, label

def load_data() -> Tuple[DataLoader, DataLoader]:
    # Dataset path
    dataset_path = os.path.join(
        'C:\\Users\\Uporabnik\\.cache\\kagglehub\\datasets\\karakaggle\\kaggle-cat-vs-dog-dataset\\versions\\1\\kagglecatsanddogs_3367a',
        'PetImages'
    )

    # Create datasets and dataloaders
    combined_dataset = CombinedDataset(dataset_path, basic_transform, advanced_transform)
    train_size = int(0.9 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_dataloader, val_dataloader