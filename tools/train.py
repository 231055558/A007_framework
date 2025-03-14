import torch
from tqdm import tqdm
import time
from tools.val import val_model, val_color_merge_model, val_output_merge_model, val_net_merge_model


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
            # 计算验证指标
            metrics = val_model(model, val_loader, metric, model_name, device)

            # 更新可视化图表
            visualizer.update_metrics(metrics)

            # 新增：将指标详细信息写入日志
            visualizer.log_metrics(metrics)

            # 保存最佳模型
            mean_acc = 0
            num_acc = 0
            for thr in metrics:
                num_acc += 1
                mean_acc += metrics[thr]["overall_accuracy"]  # 注意这里改为读取总体准确率
            if mean_acc > best_acc:
                best_acc = mean_acc
                torch.save(model.state_dict(), save_path)
                print(f'Saved best model with acc {best_acc / num_acc:.4f}')



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
            combined_inputs = torch.cat((inputs_l, inputs_r), dim=1)
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

            del outputs, loss


        epoch_loss = running_loss / len(train_loader)
        visualizer.log(f'Epoch {epoch+1}/{num_epochs} loss: {epoch_loss:.4f}')
        visualizer.update_loss(epoch_loss)

        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        # if epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     torch.save(model.state_dict(), save_path)
        #     print(f'Saved best model with loss {epoch_loss:.4f}')
        if val:
            # 计算验证指标
            metrics = val_color_merge_model(model, val_loader, metric, model_name, device)

            # 更新可视化图表
            visualizer.update_metrics(metrics)

            # 新增：将指标详细信息写入日志
            visualizer.log_metrics(metrics)

            # 保存最佳模型
            mean_acc = 0
            num_acc = 0
            for thr in metrics:
                num_acc += 1
                mean_acc += metrics[thr]["overall_accuracy"]  # 注意这里改为读取总体准确率
            if mean_acc > best_acc:
                best_acc = mean_acc
                torch.save(model.state_dict(), save_path)
                print(f'Saved best model with acc {best_acc / num_acc:.4f}')



    print("Training finished")


def train_output_merge_model(
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
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for inputs_l, inputs_r, labels in progress_bar:
            time_start = time.time()

            inputs_l = inputs_l.to(device)
            inputs_r = inputs_r.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs_l = model(inputs_l)
            outputs_r = model(inputs_r)
            outputs = outputs_l + outputs_r

            # 计算损失
            loss = loss_fn(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失和时间
            running_loss += loss.item()
            time_cost = time.time() - time_start
            progress_bar.set_postfix(loss=loss.item(), time_cost=time_cost)

            # 释放不需要的张量
            del outputs_l, outputs_r, outputs, loss

            # 清除 PyTorch 缓存
            torch.cuda.empty_cache()

        epoch_loss = running_loss / len(train_loader)
        visualizer.log(f'Epoch {epoch + 1}/{num_epochs} loss: {epoch_loss:.4f}')
        visualizer.update_loss(epoch_loss)

        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        # if epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     torch.save(model.state_dict(), save_path)
        #     print(f'Saved best model with loss {epoch_loss:.4f}')
        if val:
            # 计算验证指标
            metrics = val_output_merge_model(model, val_loader, metric, model_name, device)

            # 更新可视化图表
            visualizer.update_metrics(metrics)

            # 新增：将指标详细信息写入日志
            visualizer.log_metrics(metrics)

            # 保存最佳模型
            mean_acc = 0
            num_acc = 0
            for thr in metrics:
                num_acc += 1
                mean_acc += metrics[thr]["overall_accuracy"]  # 注意这里改为读取总体准确率
            if mean_acc > best_acc:
                best_acc = mean_acc
                torch.save(model.state_dict(), save_path)
                print(f'Saved best model with acc {best_acc / num_acc:.4f}')

    print("Training finished")


def train_net_merge_model(
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
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for inputs_l, inputs_r, labels in progress_bar:
            time_start = time.time()

            inputs_l = inputs_l.to(device)
            inputs_r = inputs_r.to(device)
            labels = labels.to(device)

            outputs = model(inputs_l, inputs_r)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            time_cost = time.time() - time_start
            progress_bar.set_postfix(loss=loss.item(), time_cost=time_cost)

        epoch_loss = running_loss / len(train_loader)
        visualizer.log(f'Epoch {epoch + 1}/{num_epochs} loss: {epoch_loss:.4f}')
        visualizer.update_loss(epoch_loss)

        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        # if epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     torch.save(model.state_dict(), save_path)
        #     print(f'Saved best model with loss {epoch_loss:.4f}')
        if val:
            # 计算验证指标
            metrics = val_net_merge_model(model, val_loader, metric, model_name, device)

            # 更新可视化图表
            visualizer.update_metrics(metrics)

            # 新增：将指标详细信息写入日志
            visualizer.log_metrics(metrics)

            # 保存最佳模型
            mean_acc = 0
            num_acc = 0
            for thr in metrics:
                num_acc += 1
                mean_acc += metrics[thr]["overall_accuracy"]  # 注意这里改为读取总体准确率
            if mean_acc > best_acc:
                best_acc = mean_acc
                torch.save(model.state_dict(), save_path)
                print(f'Saved best model with acc {best_acc / num_acc:.4f}')

    print("Training finished")