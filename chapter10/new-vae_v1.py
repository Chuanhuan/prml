
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from scipy.optimize import linear_sum_assignment
import imageio
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils

#|%%--%%| <VeA17qFsn1|tkTxO10wyH>

# ================== 配置参数 ==================
class Config:
    batch_size = 100
    latent_dim = 20
    epochs = 500
    num_classes = 10
    img_dim = 28
    initial_filters = 16
    intermediate_dim = 256
    lamb = 2.5  # 重构损失权重
    sample_std = 0.5  # 采样标准差
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#|%%--%%| <tkTxO10wyH|WvQtaplsDA>

# ================== 数据加载 ==================
def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(1, Config.img_dim, Config.img_dim))
    ])
    
    train_set = datasets.MNIST('~/Documents/data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('~/Documents/data', train=False, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=Config.batch_size)
    
    return train_loader, test_loader, train_set, test_set

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = torchvision.datasets.MNIST(root='~/Documents/data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_set = torchvision.datasets.MNIST(root='~/Documents/data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

#|%%--%%| <WvQtaplsDA|CZKH2Nh3zD>


# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#|%%--%%| <CZKH2Nh3zD|Gvpftpexin>


# Initialize the model, loss function, and optimizer
model = CNN().to(Config.device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

# Evaluation function
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Train and evaluate the model
train_model(model, train_loader, criterion, optimizer)
train_acc = evaluate(model, train_loader)
test_acc = evaluate(model, test_loader)
print(f'Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')


#|%%--%%| <Gvpftpexin|kNIHWW3wpq>


# ================== 模型定义 ==================

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Linear layer to expand 10 features to 64*7*7
        self.fc = nn.Linear(10, 64 * 7 * 7)
        # First transposed convolution: 7x7 -> 14x14
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        # Second transposed convolution: 14x14 -> 28x28
        self.deconv2 = nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1)
        # Activation
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        # Input shape: (batch, 10)
        x = self.fc(x)  # Shape: (batch, 64*7*7)
        x = x.view(-1, 64, 7, 7)  # Reshape to (batch, 64, 7, 7)
        x = self.leaky_relu(self.deconv1(x))  # Shape: (batch, 32, 14, 14)
        x = self.deconv2(x)  # Shape: (batch, 1, 28, 28)
        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64*7*7, Config.num_classes)
        self.softmax = nn.Softmax(dim=1)

        
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.softmax(x)



class ClusterVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std
        
    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)
        return {
            'recon': self.decoder(z),
            'z' : z,
            'z_mean': z_mean,
            'z_logvar': z_logvar
        }

# ================== 损失计算 ==================
def kl_divergence(p, q, eps=1e-8):
    """
    Compute the KL divergence between two images p and q.
    
    Both p and q should be tensors of the same shape, where each is 
    assumed to represent a probability distribution (i.e. non-negative and sum to 1 over the pixels).
    
    Args:
        p (Tensor): Tensor of shape (B, C, H, W) or (C, H, W) representing the first distribution.
        q (Tensor): Tensor of the same shape as p representing the second distribution.
        eps (float): A small value to avoid log(0).
    
    Returns:
        Tensor: The KL divergence for each image in the batch (if batched) or a scalar if unbatched.

        # Example usage:
# Create two dummy images that represent probability distributions
# For instance, suppose we have a batch of 10 images, each 1x28x28,
# and each image is already normalized.
    batch_size = 10
    channels = 1
    height, width = 28, 28

# Dummy distributions: here we use softmax to simulate probability distributions.
    dummy_image1 = torch.softmax(torch.randn(batch_size, channels, height, width), dim=-1)
    dummy_image2 = torch.softmax(torch.randn(batch_size, channels, height, width), dim=-1)

    kl_value = kl_divergence(dummy_image1, dummy_image2)
    print("KL divergence:", kl_value.item()). why softmax only use dim=-1)
    """
    # Add epsilon to avoid log(0)
    p = p + eps
    q = q + eps
    
    # If necessary, flatten all dimensions except batch (if batched)
    if p.dim() > 1:
        p = p.view(p.size(0), -1)
        q = q.view(q.size(0), -1)
    
    # Compute elementwise divergence and then sum over the distribution dimension.
    kl = torch.sum(p * torch.log(p / q), dim=1)  # Shape: (batch_size,)
    
    # Optionally, return the mean KL divergence over the batch
    return kl.mean()

def compute_loss(output, x, data):
    # 重构损失: MSE loss averaged over the batch
    recon_loss = 0.5 * F.mse_loss(output['recon'], x, reduction='sum') / x.size(0)
    
    # KL散度
    image1 = torch.softmax(output['z'], dim=-1)
    image2 = torch.softmax(data, dim=-1)

    kl_loss = kl_divergence(image1, image2)
    
    total_loss = Config.lamb * recon_loss + kl_loss 

    return total_loss

# ================== 训练流程 ==================
def train_model(model, train_loader, target_model):
    model.train()
    target_model.eval()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(Config.epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.epochs}')
        acc_list = []
        for data, labels in pbar:
            data = data.to(Config.device)

            labels = labels.to(Config.device)

            optimizer.zero_grad()
            
            # breakpoint()
            with torch.no_grad():
                x = target_model(data)
                _, preds_base = x.max(dim=1)
            output = model(x)

            with torch.no_grad():
                _, preds_z = target_model(output['z']).max(dim=1)
                acc = (preds_base == preds_z).float().mean().item()
                acc_list.append(acc)

            loss = compute_loss(output,x,data)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss/(pbar.n+1))
        
        acc = np.mean(acc_list)
        print(f'recon_acc: {acc:.4f}')
        print(f'Epoch {epoch+1} Average Loss: {total_loss/len(train_loader):.4f}')

# ================== 评估函数 ==================
def cluster_accuracy(y_true, y_pred):
    conf_matrix = np.zeros((Config.num_classes, Config.num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    return conf_matrix[row_ind, col_ind].sum() / len(y_true)

def evaluate(model, dataloader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc='Evaluating'):
            data = data.to(Config.device)
            output = model(data)
            preds = output['y'].argmax(dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())
    return cluster_accuracy(y_true, y_pred)

# ================== 样本生成 ==================
def cluster_sample(model, dataset, path, category=0, n=8):
    """Display real samples that are clustered into the specified category."""
    model.eval()
    indices = []
    
    loader = DataLoader(dataset, batch_size=512)
    with torch.no_grad():
        for data, _ in tqdm(loader, desc='Collecting indices'):
            data = data.to(Config.device)
            output = model(data)
            preds = output['y'].argmax(dim=1)
            batch_indices = torch.where(preds == category)[0].cpu().numpy()
            indices.extend(batch_indices)
    
    if len(indices) == 0:
        print(f"No samples found for category {category}. Skipping sample generation.")
        return
    
    figure = np.zeros((Config.img_dim*n, Config.img_dim*n))
    indices = np.random.choice(indices, n*n, replace=len(indices) < n*n)
    
    for i in range(n):
        for j in range(n):
            idx = indices[i*n + j] if i*n + j < len(indices) else 0
            digit = dataset[idx][0].squeeze().numpy()
            figure[i*Config.img_dim:(i+1)*Config.img_dim,
                   j*Config.img_dim:(j+1)*Config.img_dim] = digit
    
    imageio.imwrite(path, (figure * 255).astype(np.uint8))

def random_sample(model, path, category=0, n=8):
    """根据指定类别生成新样本"""
    model.eval()
    figure = np.zeros((Config.img_dim*n, Config.img_dim*n))
    
    with torch.no_grad():
        mean = model.gaussian.mean[category].to(Config.device)
        z = mean + torch.randn(n*n, Config.latent_dim).to(Config.device) * Config.sample_std
        samples = model.decoder(z).cpu().squeeze().numpy()
    
    for i in range(n):
        for j in range(n):
            figure[i*Config.img_dim:(i+1)*Config.img_dim,
                   j*Config.img_dim:(j+1)*Config.img_dim] = samples[i*n + j]
    
    imageio.imwrite(path, (figure * 255).astype(np.uint8))


#|%%--%%| <kNIHWW3wpq|q2ssiQTxIR>

# ================== 主程序 ==================

# train_loader, test_loader, train_set, test_set = get_dataloaders()
vae = ClusterVAE().to(Config.device)

# 训练
train_model(vae, train_loader,model)

# # 评估
# train_acc = evaluate(model, train_loader)
# test_acc = evaluate(model, test_loader)
# print(f'Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
#
#
# # # 生成样本
# if not os.path.exists('samples'):
#     os.makedirs('samples')
#
# for c in range(Config.num_classes):
#     cluster_sample(model, train_set, f'samples/cluster_{c}.png', c)
#     random_sample(model, f'samples/random_{c}.png', c)

