import torch
from timeit import default_timer as timer
from tqdm.auto import tqdm
from util.save_and_load_funs import save_checkpoint, load_checkpoint
from util.train_and_test_step_funs import train_step, test_step
from util.accuracy_fn import accuracy_fn

def train_model(
    model: torch.nn.Module, 
    train_dataloader: torch.utils.data.DataLoader, 
    val_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: str,
    writer: torch.utils.tensorboard.SummaryWriter
):
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
