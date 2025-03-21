import torch
from tqdm import tqdm
import time
from tools.val import val_double_merge_model, val_model, val_color_merge_model, val_output_merge_model, val_net_merge_model


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
    best_metrics = float('0')

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
            mean_metrics = 0
            num_thresholds = 0
            for thr in metrics:
                num_thresholds += 1
                # 计算三个指标的平均值
                current_metrics = (
                    metrics[thr]["overall_accuracy"] + 
                    metrics[thr]["overall_precision"] + 
                    metrics[thr]["overall_recall"]
                ) / 3.0
                mean_metrics += current_metrics
                
            # 计算所有阈值下的平均综合指标
            avg_metrics = mean_metrics / num_thresholds
            if avg_metrics > best_metrics:  # best_metrics需要在训练开始前初始化为0
                best_metrics = avg_metrics
                torch.save(model.state_dict(), save_path)
                print(f'Saved best model with combined metrics (ACC+PREC+REC)/3: {best_metrics:.4f}')



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
    best_metrics = float('0')

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
            mean_metrics = 0
            num_thresholds = 0
            for thr in metrics:
                num_thresholds += 1
                # 计算三个指标的平均值
                current_metrics = (
                    metrics[thr]["overall_accuracy"] + 
                    metrics[thr]["overall_precision"] + 
                    metrics[thr]["overall_recall"]
                ) / 3.0
                mean_metrics += current_metrics
                
            # 计算所有阈值下的平均综合指标
            avg_metrics = mean_metrics / num_thresholds
            if avg_metrics > best_metrics:  # best_metrics需要在训练开始前初始化为0
                best_metrics = avg_metrics
                torch.save(model.state_dict(), save_path)
                print(f'Saved best model with combined metrics (ACC+PREC+REC)/3: {best_metrics:.4f}')



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
    best_metrics = float('0')

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
            mean_metrics = 0
            num_thresholds = 0
            for thr in metrics:
                num_thresholds += 1
                # 计算三个指标的平均值
                current_metrics = (
                    metrics[thr]["overall_accuracy"] + 
                    metrics[thr]["overall_precision"] + 
                    metrics[thr]["overall_recall"]
                ) / 3.0
                mean_metrics += current_metrics
                
            # 计算所有阈值下的平均综合指标
            avg_metrics = mean_metrics / num_thresholds
            if avg_metrics > best_metrics:  # best_metrics需要在训练开始前初始化为0
                best_metrics = avg_metrics
                torch.save(model.state_dict(), save_path)
                print(f'Saved best model with combined metrics (ACC+PREC+REC)/3: {best_metrics:.4f}')

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
    best_metrics = float('0')

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
            mean_metrics = 0
            num_thresholds = 0
            for thr in metrics:
                num_thresholds += 1
                # 计算三个指标的平均值
                current_metrics = (
                    metrics[thr]["overall_accuracy"] + 
                    metrics[thr]["overall_precision"] + 
                    metrics[thr]["overall_recall"]
                ) / 3.0
                mean_metrics += current_metrics
                
            # 计算所有阈值下的平均综合指标
            avg_metrics = mean_metrics / num_thresholds
            if avg_metrics > best_metrics:  # best_metrics需要在训练开始前初始化为0
                best_metrics = avg_metrics
                torch.save(model.state_dict(), save_path)
                print(f'Saved best model with combined metrics (ACC+PREC+REC)/3: {best_metrics:.4f}')

    print("Training finished")


def train_double_merge_model(
        model_1,
        model_2,
        head,
        train_loader,
        loss_fn,
        optimizer_1,
        optimizer_2,
        visualizer,
        device='cuda',
        model_name="default_model",
        num_epochs=100,
        save_path='best',
        val=False,
        val_loader=None,
        metric=None
):
    visualizer.log("-----------start training--------------")
    if val:
        assert val_loader is not None and metric is not None
    model_1.to(device)
    model_2.to(device)
    head.to(device)
    best_metrics = float('0')

    for epoch in range(num_epochs):
        model_1.train()
        model_2.train()
        head.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for inputs_l, inputs_r, labels in progress_bar:
            time_start = time.time()

            inputs_l = inputs_l.to(device)
            inputs_r = inputs_r.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs_l = model_1(inputs_l)
            outputs_r = model_2(inputs_r)
            output_linear = torch.cat((outputs_l, outputs_r), dim=1)
            outputs = head(output_linear)

            # 计算损失
            loss = loss_fn(outputs, labels)

            # 反向传播
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            loss.backward()
            optimizer_1.step()
            optimizer_2.step()

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
            metrics = val_double_merge_model(model_1, model_2, head, val_loader, metric, model_name, device)

            # 更新可视化图表
            visualizer.update_metrics(metrics)

            # 新增：将指标详细信息写入日志
            visualizer.log_metrics(metrics)

            # 保存最佳模型
            mean_metrics = 0
            num_thresholds = 0
            for thr in metrics:
                num_thresholds += 1
                # 计算三个指标的平均值
                current_metrics = (
                    metrics[thr]["overall_accuracy"] + 
                    metrics[thr]["overall_precision"] + 
                    metrics[thr]["overall_recall"]
                ) / 3.0
                mean_metrics += current_metrics
                
            # 计算所有阈值下的平均综合指标
            avg_metrics = mean_metrics / num_thresholds
            if avg_metrics > best_metrics:  # best_metrics需要在训练开始前初始化为0
                best_metrics = avg_metrics
                torch.save(model_1.state_dict(), save_path + "_model_1.pth")
                torch.save(model_2.state_dict(), save_path + "_model_2.pth")
                torch.save(head.state_dict(), save_path + "_head.pth")
                print(f'Saved best model with combined metrics (ACC+PREC+REC)/3: {best_metrics:.4f}')

    print("Training finished")

