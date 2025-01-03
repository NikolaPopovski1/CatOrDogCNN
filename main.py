##################################################################################################################
##################################################################################################################
### FULL PYTORCH CODE AND !!TRAINING/LOADING OF MODEL AT THE END OF THE CODE #####################################
##################################################################################################################
##################################################################################################################


######################################
######################## Load data ###
######################################
import torch
import torch.nn as nn
from util.load_data import CombinedDataset, basic_transform, advanced_transform, load_data
from DogOrCatModel import DogOrCatModel
from torch.utils.data import DataLoader

torch.manual_seed(442)
train_dataloader, val_dataloader = load_data()
model = DogOrCatModel(input_shape=3, hidden_units=16, output_shape=2)
images = torch.randn(size=(1, 3, 64, 64))


######################################
########### Optimizer, loss, CUDA? ###
######################################
torch.manual_seed(42)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
device = "cuda" if torch.cuda.is_available() else "cpu"


######################################
####################### Tensorflow ###
######################################
torch.manual_seed(442)
from torch.utils.tensorboard import SummaryWriter
import os
# Set the log directory for TensorBoard
log_dir = os.path.join(os.getcwd(), 'runs')  # Full path to the 'runs' folder
writer = SummaryWriter(log_dir=log_dir)


######################################
### Training and testing functions ###
######################################
from util.train_model import train_model
train_model(
    model=model, 
    train_dataloader=train_dataloader, 
    val_dataloader=val_dataloader,
    loss_fn=loss_fn, 
    optimizer=optimizer, 
    device=device,
    writer=writer
)

######################################
##### Test the model on single img ###
######################################
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
torch.manual_seed(442)

image_path = "test_imgs/cat.jpg"
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


######################################
################### Evaluate model ###
######################################
import util.eval_model as em
from util.accuracy_fn import accuracy_fn

model_0_results = em.eval_model(
    model=model, 
    data_loader=val_dataloader,
    loss_fn=loss_fn, 
    accuracy_fn=accuracy_fn
)
print(model_0_results)