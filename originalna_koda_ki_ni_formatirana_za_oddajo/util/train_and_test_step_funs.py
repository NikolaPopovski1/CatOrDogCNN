# train and test step
import torch

def train_step(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device,
               epoch: int = 0,
               val_dataloader: torch.utils.data.DataLoader = None,
               writer: torch.utils.tensorboard.SummaryWriter = None):
    train_loss, train_acc = 0, 0
    model.train()
    for batch, (X, y) in enumerate(data_loader): # Iti skozi vse batche
        X = X.float()
        # Podatke na napravo
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)
        
        # Loss
        loss = loss_fn(y_pred, y)
         # Dodaj loss v train_loss
        train_loss += loss.item()
        # Kalkulacija točnosti
        train_acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1))

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()
    
    # Loss avg za batch
    train_loss /= len(val_dataloader)
    writer.add_scalar("Loss/train", train_loss, epoch)
    # Točnost avg za batch
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    
    print(f"\nTrain loss: {train_loss:.5f} | Training Acc: {train_acc:.2f}%")


def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device,
              epoch: int = 0,
              writer: torch.utils.tensorboard.SummaryWriter = None):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X = X.float()
            X, y = X.to(device), y.to(device)

            test_pred = model(X)

            test_loss += loss_fn(test_pred, y).item()
            test_acc += accuracy_fn(y_true=y, 
                                    y_pred=test_pred.argmax(dim=1))
        
        test_loss /= len(data_loader)
        writer.add_scalar("Loss/test", test_loss, epoch)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}%")