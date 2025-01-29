import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
import imageio

# 超参数与设备配置
batch_size = 100
latent_dim = 20
epochs = 50
num_classes = 10
img_dim = 28
initial_filters = 16
intermediate_dim = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(1, img_dim, img_dim))]
)

train_dataset = datasets.MNIST(
    "~/Documents/data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST("~/Documents/data", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class GaussianLayer(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.mean = nn.Parameter(torch.zeros(num_classes, latent_dim))

    def forward(self, z):
        # 输入z形状: (batch_size, latent_dim)
        # 输出形状: (batch_size, num_classes, latent_dim)
        return z.unsqueeze(1) - self.mean.unsqueeze(0)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # 第一组卷积
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # 第二组卷积
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)  # 输出形状: (batch, 64, 7, 7)
        x = self.flatten(x)  # 形状: (batch, 64*7*7)
        return self.fc_mean(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.conv_layers = nn.Sequential(
            # 第一组转置卷积
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # 第二组转置卷积
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # 第三组转置卷积
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 64, 7, 7)  # 重塑为卷积输入形状
        return self.conv_layers(z)  # 输出形状: (batch, 1, 28, 28)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, z):
        return self.net(z)


class ClusterVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.classifier = Classifier()
        self.gaussian = GaussianLayer(num_classes, latent_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        # 编码
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)

        # 解码
        x_recon = self.decoder(z)

        # 分类
        y = self.classifier(z)

        # 高斯层处理
        z_prior_mean = self.gaussian(z)  # 形状: (batch, num_classes, latent_dim)

        return x_recon, z_prior_mean, y, z_mean, z_logvar


model = ClusterVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练循环
lamb = 2.5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        # 前向传播
        x_recon, z_prior_mean, y, z_mean, z_logvar = model(data)

        # 重构损失 (形状对齐)
        recon_loss = 0.5 * F.mse_loss(x_recon, data, reduction="sum") / data.size(0)

        # KL散度损失 (维度对齐)
        z_mean_exp = z_mean.unsqueeze(1)  # (batch, 1, latent_dim)
        z_logvar_exp = z_logvar.unsqueeze(1)  # (batch, 1, latent_dim)

        # 计算KL项: -0.5 * (logvar - (z_prior_mean)^2)
        kl_elements = -0.5 * (z_logvar_exp - z_prior_mean.pow(2))

        # 使用einsum进行张量乘积 (batch, num_classes) x (batch, num_classes, latent_dim)
        kl_loss = torch.einsum("bi,bil->", y, kl_elements).mean()

        # 分类熵损失
        cat_loss = -(y * torch.log(y + 1e-8)).sum(dim=1).mean()

        # 总损失
        loss = lamb * recon_loss + kl_loss + cat_loss

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")


# 评估函数
def cluster_accuracy(y_true, y_pred):
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    return conf_matrix[row_ind, col_ind].sum() / len(y_true)


def evaluate(model, dataloader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            z_mean, _ = model.encoder(data)
            y_probs = model.classifier(z_mean)
            y_pred.extend(torch.argmax(y_probs, dim=1).cpu().numpy())
            y_true.extend(labels.numpy())
    return cluster_accuracy(y_true, y_pred)


# 计算准确率
train_acc = evaluate(model, train_loader)
test_acc = evaluate(model, test_loader)
print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")


# 生成样本函数
def generate_samples(model, path, category=0, num_samples=64):
    model.eval()
    with torch.no_grad():
        # 获取类别中心
        mean = model.gaussian.mean[category].unsqueeze(0).to(device)

        # 生成样本
        samples = []
        for _ in range(num_samples):
            z = mean + torch.randn(1, latent_dim).to(device)
            sample = model.decoder(z).cpu().squeeze().numpy()
            samples.append(sample)

        # 创建网格
        grid = np.zeros((img_dim * 8, img_dim * 8))
        for i in range(8):
            for j in range(8):
                grid[
                    i * img_dim : (i + 1) * img_dim, j * img_dim : (j + 1) * img_dim
                ] = samples[i * 8 + j]

        imageio.imwrite(path, (grid * 255).astype(np.uint8))


# 保存样本
if not os.path.exists("samples"):
    os.makedirs("samples")

for c in range(num_classes):
    generate_samples(model, f"samples/class_{c}_samples.png", category=c)
