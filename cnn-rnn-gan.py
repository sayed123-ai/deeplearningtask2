#CNN

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

from torch.utils.data import Subset
trainset = Subset(trainset, range(10000))  # faster

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Simple CNN
class SimpleCNN(nn.Module):
    def _init_(self):
        super()._init_()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# Train
def train(model, epochs):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        losses.append(total_loss/len(trainloader))
        print(f"Epoch {epoch+1}, Loss: {losses[-1]:.4f}")

    return losses

# Test
def test(model):
    correct = 0
    total = 0

    preds = []
    labels = []

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, p = torch.max(out, 1)

            correct += (p == y).sum().item()
            total += y.size(0)

            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())

    print("Accuracy:", 100*correct/total)
    print("Confusion Matrix:\n", confusion_matrix(labels, preds))

# Run CNN
print("=== SIMPLE CNN ===")
cnn = SimpleCNN()
cnn_loss = train(cnn, 3)
test(cnn)

plt.plot(cnn_loss)
plt.title("CNN Loss")
plt.show()


# RNN 

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy sequence data
X = torch.randint(0, 100, (200, 20)).to(device)
y = torch.randint(0, 2, (200,)).float().to(device)

# Model
class RNNModel(nn.Module):
    def _init_(self, type):
        super()._init_()
        self.embedding = nn.Embedding(100, 32)

        if type == "LSTM":
            self.rnn = nn.LSTM(32, 32, batch_first=True)
        elif type == "GRU":
            self.rnn = nn.GRU(32, 32, batch_first=True)
        else:
            self.rnn = nn.RNN(32, 32, batch_first=True)

        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return torch.sigmoid(self.fc(out))

# Train
def train_model(name):
    model = RNNModel(name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    losses = []

    for epoch in range(5):
        out = model(X).squeeze()
        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"{name} Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return losses

# Run
rnn_loss = train_model("RNN")
lstm_loss = train_model("LSTM")
gru_loss = train_model("GRU")

# Plot
plt.plot(rnn_loss, label="RNN")
plt.plot(lstm_loss, label="LSTM")
plt.plot(gru_loss, label="GRU")
plt.legend()
plt.title("RNN Comparison")
plt.show()


# GAN

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Generator
class Generator(nn.Module):
    def _init_(self):
        super()._init_()
        self.net = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x).view(-1,1,28,28)

# Discriminator
class Discriminator(nn.Module):
    def _init_(self):
        super()._init_()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

G = Generator().to(device)
D = Discriminator().to(device)

loss_fn = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=0.0002)
opt_D = optim.Adam(D.parameters(), lr=0.0002)

g_loss_list = []
d_loss_list = []

# Train
for epoch in range(5):
    for real, _ in loader:
        real = real.to(device)
        bs = real.size(0)

        real_label = torch.ones(bs,1).to(device)
        fake_label = torch.zeros(bs,1).to(device)

        # Train D
        noise = torch.randn(bs,100).to(device)
        fake = G(noise)

        loss_real = loss_fn(D(real), real_label)
        loss_fake = loss_fn(D(fake.detach()), fake_label)
        d_loss = loss_real + loss_fake

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # Train G
        loss_G = loss_fn(D(fake), real_label)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    g_loss_list.append(loss_G.item())
    d_loss_list.append(d_loss.item())

    print(f"Epoch {epoch+1}, G Loss: {loss_G.item():.4f}, D Loss: {d_loss.item():.4f}")

    # Show images
    sample = G(torch.randn(16,100).to(device)).cpu()
    grid = torchvision.utils.make_grid(sample, nrow=4, normalize=True)
    plt.imshow(grid.permute(1,2,0))
    plt.title(f"Epoch {epoch+1}")
    plt.axis("off")
    plt.show()

# Plot losses
plt.plot(g_loss_list, label="Generator")
plt.plot(d_loss_list, label="Discriminator")
plt.legend()
plt.title("GAN Loss")
plt.show()

