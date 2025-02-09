import torch
from tqdm import tqdm
import time
from tools.val import val_model
def train_model(
        model,
        train_loader,
        loss_fn,
        optimizer,
        device='cuda',
        num_epochs=100,
        save_path='best_model.pth',
        val=False,
        val_loader=None,
        metric=None,
):
    if val:
        assert val_loader is not None and metric is not None
    model.to(device)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for inputs, labels in progress_bar:
            time_start = time.time()
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            time_cost = time.time() - time_start
            progress_bar.set_postfix(loss=loss.item(), time_cost=time_cost)


        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f'Saved best model with loss {epoch_loss:.4f}')
        if val is True:
            metrics = val_model(
                model, val_loader, metric, device
            )




    print("Training finished")
