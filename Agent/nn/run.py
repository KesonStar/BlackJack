import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from util import data_preprocessing, set_training_params
from blackjack_nn import BlackjackNN

def train(model, train_dataset, val_dataset, criterion, optimizer, 
                epochs=50, batch_size=32, model_path="blackjack_nn_model.pth"):
    """
    使用给定的优化器和损失函数训练模型。

    Args:
        model (nn.Module): 待训练的模型。
        train_dataset (Dataset): 训练集。
        val_dataset (Dataset): 验证集。
        criterion (nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        epochs (int): 训练轮数，默认为 20。
        batch_size (int): 批大小，默认为 32。
        model_path (str): 保存模型的路径，默认为 "blackjack_nn_model.pth"。

    Returns:
        None
    """
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Training on {device}...")
    for epoch in tqdm(range(epochs)):
        model.train()  
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

     
        avg_train_loss = running_loss / len(train_loader)
        if (epoch+1) % 5 == 0 and epoch != 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

       
        if (epoch+1) % 10 == 0 and epoch != 0:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total
            print(f"Epoch [{epoch + 1}/{epochs}], Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")

    # 保存模型参数
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    
if __name__ == '__main__':
    X, y = data_preprocessing("data/blackjack_simulation_data.csv")
    train_X_split, val_X_split, train_y_split, val_y_split = train_test_split(
    X, y, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(train_X_split, train_y_split)
    val_dataset = TensorDataset(val_X_split, val_y_split)
    
    model = BlackjackNN()
    loss, optimizer = set_training_params(model)
    train(model, train_dataset, val_dataset, loss, optimizer, epochs=20, batch_size=32, model_path="blackjack_nn_model.pth")