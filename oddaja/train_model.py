import torch
from timeit import default_timer as timer
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


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

def train_model(
    model: torch.nn.Module, 
    train_dataloader: torch.utils.data.DataLoader, 
    val_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: str
):
    torch.manual_seed(442)
    from torch.utils.tensorboard import SummaryWriter
    import os
    # Set the log directory for TensorBoard
    log_dir = os.path.join(os.getcwd(), 'runs')  # Full path to the 'runs' folder
    writer = SummaryWriter(log_dir=log_dir)

    train_time_start = timer()
    epochs = 1
    
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
            save_checkpoint(checkpoint, filename=f"trained_model_epoch_{epoch+1}.pth.tar")
        
        writer.flush()

    # Calculate and print training time
    train_time_end = timer()
    print(f"Training time: {train_time_end - train_time_start:.2f}s")
    writer.close()
