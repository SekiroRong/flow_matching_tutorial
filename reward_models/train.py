import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
from tqdm import tqdm
from reward_model import MNIST_CNN

# ========== 1. 配置参数+创建保存权重的文件夹（你之前问的os.makedirs） ==========
os.makedirs("./mnist_pretrained", exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
epochs = 5
lr = 1e-3

# ========== 2. MNIST数据加载（官方标准预处理，无需修改） ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST官方均值/方差，固定值！
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = MNIST_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ========== 4. 训练+保存权重（5个epoch足够，收敛完毕） ==========
def train(model, loader, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data, target in tqdm(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    # 保存训练好的权重到本地（核心！这就是你的MNIST预训练权重）
    torch.save(model.state_dict(), "./mnist_pretrained/mnist_cnn_993.pth")
    print("✅ 权重保存完成：./mnist_pretrained/mnist_cnn_993.pth")

# ========== 5. 测试准确率 ==========
def test(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = correct / len(loader.dataset)
    print(f"✅ 测试集准确率: {acc:.4f} ({correct}/{len(loader.dataset)})")

# ========== 一键运行 ==========
train(model, train_loader, epochs)
test(model, test_loader)