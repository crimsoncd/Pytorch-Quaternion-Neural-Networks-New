import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from quaternion_layers import QuaternionConv
from quaternion_ops import get_modulus

class QMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.qconv1 = QuaternionConv(4, 32, kernel_size=3, stride=1, padding=1)
        self.qconv2 = QuaternionConv(32, 64, kernel_size=3, stride=2, padding=1)
        self.qconv3 = QuaternionConv(64, 64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.act(self.qconv1(x))
        x = self.act(self.qconv2(x))
        x = self.act(self.qconv3(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = get_modulus(x, vector_form=True)
        return self.fc(x)

def gray_to_quaternion(t):
    r = torch.zeros_like(t)
    return torch.cat([r, t, t, t], dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([T.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

model = QMNIST(num_classes=10).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for imgs, labels in trainloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        qimgs = gray_to_quaternion(imgs)
        logits = model(qimgs)
        loss = crit(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in testloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        qimgs = gray_to_quaternion(imgs)
        logits = model(qimgs)
        preds = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
print("Test Acc:", correct / total)