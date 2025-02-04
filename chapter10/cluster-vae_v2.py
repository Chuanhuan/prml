import torch
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

# ================== 配置参数 ==================
class Config:
    batch_size = 100
    latent_dim = 20
    epochs = 10
    num_classes = 10
    img_dim = 28
    initial_filters = 16
    intermediate_dim = 256
    lamb = 2.5  # 重构损失权重
    sample_std = 0.5  # 采样标准差
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ================== 模型定义 ==================
class GaussianLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(Config.num_classes, Config.latent_dim))
        
    def forward(self, z):
        # return z.unsqueeze(1) - self.mean.unsqueeze(0)
        return z.unsqueeze(1) 

class Encoder(nn.Module):
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
        self.fc_mean = nn.Linear(64*7*7, Config.latent_dim)
        self.fc_logvar = nn.Linear(64*7*7, Config.latent_dim)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc_mean(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(Config.latent_dim, 64*7*7)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 64, 7, 7)
        return self.conv(z)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Config.latent_dim, Config.intermediate_dim),
            nn.ReLU(),
            nn.Linear(Config.intermediate_dim, Config.num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, z):
        return self.net(z)

class ClusterVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.classifier = Classifier()
        self.gaussian = GaussianLayer()
        
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std
        
    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)
        return {
            'recon': self.decoder(z),
            'z_prior': self.gaussian(z),
            'y': self.classifier(z), # p(y|x)
            'z_mean': z_mean,
            'z_logvar': z_logvar
        }

# ================== 损失计算 ==================
# def compute_loss(output, x):
#     # 重构损失
#     recon_loss = 0.5 * F.mse_loss(output['recon'], x, reduction='sum') / x.size(0)
#
#     # KL散度
#     z_logvar = output['z_logvar'].unsqueeze(1)
#     kl_elements = -0.5 * (z_logvar - output['z_prior'].pow(2))
#
#     # 方法1:
#     kl_loss = torch.einsum('bi,bil->b', output['y'], kl_elements).mean()
#
#     # 方法2: 修正后的替代方案
#     # weighted_kl = output['y'].unsqueeze(-1) * kl_elements  # (batch, num_classes, latent_dim)
#     # kl_loss = weighted_kl.sum(dim=(1,2)).mean()  # 标量
#
#     # 分类熵
#     cat_loss = -(output['y'] * torch.log(output['y'] + 1e-8)).sum(1).mean()
#     print(f'y: {output["y"].argmax(dim=1)}')
#
#     return Config.lamb * recon_loss + kl_loss + cat_loss

def compute_loss(output, x, labels):
    # 重构损失: MSE loss averaged over the batch
    recon_loss = 0.5 * F.mse_loss(output['recon'], x, reduction='sum') / x.size(0)
    
    # KL散度
    # Expand z_logvar to have a new dimension for the num_classes (if needed)
    z_logvar = output['z_logvar'].unsqueeze(1)  # shape: (batch, 1, latent_dim)
    # Here, we assume output['z_prior'] is of shape (batch, num_classes, latent_dim)
    # and compute elementwise: -0.5 * (z_logvar - (z_prior)^2)
    kl_elements = -0.5 * (z_logvar - output['z_prior'].pow(2))
    
    # Using einsum to mimic K.batch_dot(K.expand_dims(y,1), kl_elements)
    kl_loss = torch.einsum('bi,bil->b', output['y'], kl_elements).mean()
    
    # 分类熵（原始正则项）
    cat_loss = -(output['y'] * torch.log(output['y'] + 1e-8)).sum(1).mean()
    
    # 新增：分类损失：交叉熵损失 (由于 classifier 已经输出概率, 使用 nll_loss 需取对数)
    classification_loss = F.nll_loss(torch.log(output['y'] + 1e-8), labels)
    total_loss = Config.lamb * recon_loss + kl_loss + cat_loss 
    total_loss *=  classification_loss
    print(f'y: {output["y"].argmax(dim=1)}')

    return total_loss

# ================== 训练流程 ==================
def train_model(model, train_loader):
    model.train()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(Config.epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.epochs}')
        for data, labels in pbar:
            data = data.to(Config.device)
            labels = labels.to(Config.device)
            optimizer.zero_grad()
            
            output = model(data)
            # loss = compute_loss(output, data)
            loss = compute_loss(output, data,labels)
            print(f'Loss: {loss.item()}')
            if output['y'].argmax(dim=1).float().mean() == output['y'].argmax(dim=1)[0]:
                print(f'Loss: {loss.item()}')
                break
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss/(pbar.n+1))
            
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

def plot_dataloader_samples(dataloader, path, n=8):
    """Plot samples from dataloader"""
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    grid_img = vutils.make_grid(real_batch[0][:n*n], padding=2, normalize=True)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig(path)
    plt.close()


# ================== 主程序 ==================
if __name__ == "__main__":
    # 初始化
    train_loader, test_loader, train_set, test_set = get_dataloaders()
    model = ClusterVAE().to(Config.device)
    
    # 训练
    train_model(model, train_loader)
    
    # # 评估
    train_acc = evaluate(model, train_loader)
    test_acc = evaluate(model, test_loader)
    print(f'Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
    

    # plot_dataloader_samples(train_loader, 'samples/train_samples.png')
    #
    # for c in range(Config.num_classes):
    #     cluster_sample(model, train_set, f'samples/cluster_{c}.png', c)
    #     random_sample(model, f'samples/random_{c}.png', c)
    #
    # # 生成样本
    # if not os.path.exists('samples'):
    #     os.makedirs('samples')
    #
    # for c in range(Config.num_classes):
    #     cluster_sample(model, train_set, f'samples/cluster_{c}.png', c)
    #     random_sample(model, f'samples/random_{c}.png', c)
