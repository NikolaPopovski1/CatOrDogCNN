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
import torch
from timeit import default_timer as timer
from tqdm.auto import tqdm
from util.save_and_load_funs import save_checkpoint, load_checkpoint
from util.train_and_test_step_funs import train_step, test_step
from util.accuracy_fn import accuracy_fn

torch.manual_seed(442)

train_time_start = timer()
epochs = 12
load_model = True

if load_model:
    try:
        checkpoint = torch.load("checkpoint_12.pth.tar", weights_only=False)
        load_checkpoint(checkpoint, model, optimizer)
    except FileNotFoundError:
        print("Checkpoint file not found. Starting from scratch.")
else:
    # Training loop
    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch}\n-------------------------------")

        # Perform training step and get the training loss
        train_loss = train_step(
            model=model, 
            data_loader=train_dataloader, 
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device,
            val_dataloader=val_dataloader,
            epoch=epoch,
            writer=writer
        )
        
        # Perform testing step and get the test loss
        test_loss = test_step(
            model=model,
            data_loader=val_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device,
            epoch=epoch,
            writer=writer
        )
        
        checkpoint = {
            'state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict()
        }
        if (epoch+1) >= 11:
            save_checkpoint(checkpoint, filename=f"checkpoint_{epoch+1}.pth.tar")
        
        writer.flush()

    # Calculate and print training time
    train_time_end = timer()
    print(f"Training time: {train_time_end - train_time_start:.2f}s")
    writer.close()


######################################
##### Test the model on single img ###
######################################
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
torch.manual_seed(442)

image_path = "test_imgs/8137.jpg"
image = torchvision.io.read_image(image_path).type(torch.float32) / 255.0

min_dim = min(image.shape[1], image.shape[2])  # image.shape is (C, H, W)
image_transform = transforms.Compose([
    transforms.CenterCrop(min_dim),  # Crop to the smallest dimension (1:1 ratio)
    transforms.Resize((64, 64))      # Resize to 64x64
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
##### Test the model on single img ###
######################################
import util.eval_model as em

model_0_results = em.eval_model(
    model=model, 
    data_loader=val_dataloader,
    loss_fn=loss_fn, 
    accuracy_fn=accuracy_fn
)
print(model_0_results)