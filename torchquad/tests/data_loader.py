import sys
sys.path.append("../")

import torch
from torchvision import datasets
from torchquad import set_up_backend


set_up_backend('torch')
assert(torch.cuda_available())

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

train_loader = DataLoader(training_data, batch_size=64, shuffle=True)

for (i, x) in enumerate(train_loader):
    pass
