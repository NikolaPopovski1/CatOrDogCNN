##################################################################################################################
##################################################################################################################
### FULL PYTORCH CODE AND !!TRAINING/LOADING OF MODEL AT THE END OF THE CODE #####################################
##################################################################################################################
##################################################################################################################


############################################################################
############## Load data and model #########################################
############################################################################
import torch
import torch.nn as nn
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

# Albumentations transform
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
# Filtrirana slikaMapa za obdelavo neveljavnih datotek
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
# Kombinirani nabor podatkov za osnovne in napredne transformacije
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
class DogOrCatModel(nn.Module):
    def __init__(self, 
                 input_shape: int, 
                 hidden_units: int, 
                 output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3,
                      stride=1,
                      padding=1),
            #nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units), # samo normalizira, ne spreminja oblike. Ni nekaj pomembnega, lahko izpustimo samo I think da bi bli podatki slabši oz. tak je razloženo, nimam časa naštudirati njegove podrobnosti
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        second_hidden_units = hidden_units * 2
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=second_hidden_units, 
                      kernel_size=3,
                      stride=1,
                      padding=1),
            #nn.BatchNorm2d(second_hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=second_hidden_units, 
                      out_channels=second_hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(second_hidden_units), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        third_hidden_units = hidden_units * 4
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=second_hidden_units, 
                      out_channels=third_hidden_units, 
                      kernel_size=3,
                      stride=1,
                      padding=1),
            #nn.BatchNorm2d(third_hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=third_hidden_units, 
                      out_channels=third_hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(third_hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Fully connected layer; in_features is calculated dynamically
            nn.Linear(in_features=hidden_units * 4 * 8 * 8,# na začetku dat na hidden_units*0, da najdemo napake kar se tiče dimenzij, potem pa spremenimo
                      out_features=128), 
            nn.Tanh(),
            nn.Linear(128, output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        #print(f"E: {x.shape}")
        x = self.block_2(x)
        #print(f"E: {x.shape}")
        x = self.block_3(x)
        #print(f"E: {x.shape}")
        x = self.classifier(x)
        return x

torch.manual_seed(442)
train_dataloader, val_dataloader = load_data()
model = DogOrCatModel(input_shape=3, hidden_units=16, output_shape=2)
images = torch.randn(size=(1, 3, 64, 64))


############################################################################
########### Optimizer, loss, CUDA? #########################################
############################################################################
torch.manual_seed(42)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
device = "cuda" if torch.cuda.is_available() else "cpu"

############################################################################
####################### Load model #########################################
############################################################################
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])

try:
    checkpoint = torch.load("trained_model_epoch_12.pth.tar", weights_only=False)
    load_checkpoint(checkpoint, model, optimizer)
except FileNotFoundError:
    print("Checkpoint file not found. Starting from scratch.")


############################################################################
###################### Train model #########################################
############################################################################
"""
from train_model import train_model
train_model(
    model=model, 
    train_dataloader=train_dataloader, 
    val_dataloader=val_dataloader,
    loss_fn=loss_fn, 
    optimizer=optimizer, 
    device=device
)
"""


############################################################################
##### Test the model on single img #########################################
############################################################################
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
torch.manual_seed(442)

image_path = "dog.jpg"
image = torchvision.io.read_image(image_path).type(torch.float32) / 255.0

min_dim = min(image.shape[1], image.shape[2])  # image.shape je (C, H, W)
image_transform = transforms.Compose([
    transforms.CenterCrop(min_dim),
    transforms.Resize((64, 64))
])
image = image_transform(image)

with torch.inference_mode():
    custom_image_prediction = model(image.unsqueeze(0).to(device)) # Add batch dimension, make sure it's on the right device

#"""
# show the image with the label and no grid
def show_image(image, label, classes):
    plt.imshow(image.permute(1, 2, 0))
    plt.title(classes[label])
    plt.axis("off")
    plt.show()

# Show the image with the prediction
show_image(image, custom_image_prediction.argmax(dim=1).item(), ["Cat", "Dog"])
#"""