from torch import nn
import torch
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