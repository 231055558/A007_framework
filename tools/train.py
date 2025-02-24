import torch
from tqdm import tqdm
import time
from tools.val import val_model
def train_model(
        model,
        train_loader,
        loss_fn,
        optimizer,
        visualizer,
        device='cuda',
        model_name="default_model",
        num_epochs=100,
        save_path='best_model.pth',
        val=False,
        val_loader=None,
        metric=None
):
    visualizer.log("-----------start training--------------")
    if val:
        assert val_loader is not None and metric is not None
    model.to(device)
    best_acc = float('0')

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
        visualizer.log(f'Epoch {epoch+1}/{num_epochs} loss: {epoch_loss:.4f}')
        visualizer.update_loss(epoch_loss)

        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        # if epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     torch.save(model.state_dict(), save_path)
        #     print(f'Saved best model with loss {epoch_loss:.4f}')
        if val:
            metrics = val_model(
                model, val_loader, metric, model_name, device
            )
            visualizer.update_metrics(metrics)

            mean_acc = 0
            num_acc = 0
            for thr in metrics:
                num_acc += 1
                mean_acc += metrics[thr]["accuracy"]
            if mean_acc > best_acc:
                best_acc = mean_acc
                torch.save(model.state_dict(), save_path)
                print(f'Saved best model with acc {best_acc/num_acc:.4f}')



    print("Training finished")

def train_color_merge_model(
        model,
        train_loader,
        loss_fn,
        optimizer,
        visualizer,
        device='cuda',
        model_name="default_model",
        num_epochs=100,
        save_path='best_model.pth',
        val=False,
        val_loader=None,
        metric=None
):
    visualizer.log("-----------start training--------------")
    if val:
        assert val_loader is not None and metric is not None
    model.to(device)
    best_acc = float('0')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for inputs_l, inputs_r, labels in progress_bar:
            time_start = time.time()
            combined_inputs = torch.cat((inputs_l, inputs_r), dim=0)
            inputs = combined_inputs.to(device)
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
        visualizer.log(f'Epoch {epoch+1}/{num_epochs} loss: {epoch_loss:.4f}')
        visualizer.update_loss(epoch_loss)

        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        # if epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     torch.save(model.state_dict(), save_path)
        #     print(f'Saved best model with loss {epoch_loss:.4f}')
        if val:
            metrics = val_model(
                model, val_loader, metric, model_name, device
            )
            visualizer.update_metrics(metrics)

            mean_acc = 0
            num_acc = 0
            for thr in metrics:
                num_acc += 1
                mean_acc += metrics[thr]["accuracy"]
            if mean_acc > best_acc:
                best_acc = mean_acc
                torch.save(model.state_dict(), save_path)
                print(f'Saved best model with acc {best_acc/num_acc:.4f}')



    print("Training finished")
