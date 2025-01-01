import csv
import time
import torch
import random
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import StepLR
from spikingjelly.activation_based import layer, neuron, encoding, functional, learning

random.seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

train_mnist = datasets.MNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())
test_mnist = datasets.MNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_mnist, batch_size=64, shuffle=True)
test_loader = DataLoader(test_mnist, batch_size=64, shuffle=True)
# Create data loaders

T = 30
lr = 0.001
tau_pre = 10.
tau_post = 10.
step_mode = 'm'
# Define parameters

def f_weight(x):
    return torch.clamp(x, -1, 1.)
    # Clamp STDP weights between -1 and 1

net = nn.Sequential(
    layer.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
    neuron.IFNode(),
    layer.MaxPool2d(2, 2),
    layer.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
    neuron.IFNode(),
    layer.MaxPool2d(2, 2),
    layer.Flatten(),
    layer.Linear(16 * 7 * 7, 64, bias=False),
    neuron.IFNode(),
    layer.Linear(64, 10, bias=False),
    neuron.IFNode(),
).to(device)
# Define the propagation network

functional.set_step_mode(net, step_mode)
# Define multi-step mode

instances_stdp = (layer.Conv2d, )
# Define layers that require STDP adjustment

stdp_learners = []

for i in range(net.__len__()):
    if isinstance(net[i], instances_stdp):
        stdp_learners.append(
            learning.STDPLearner(
                step_mode=step_mode,
                synapse=net[i],
                sn=net[i+1],
                tau_pre=tau_pre,
                tau_post=tau_post,
                f_pre=f_weight,
                f_post=f_weight
            )
        )
# Iterate through network layers, find layers requiring STDP connections, add them to STDP learners, and define parameters connected to the next layer

params_stdp = []
for m in net.modules():
    if isinstance(m, instances_stdp):
        for p in m.parameters():
            params_stdp.append(p)

params_stdp_set = set(params_stdp)
params_gradient_descent = []
for p in net.parameters():
    if p not in params_stdp_set:
        params_gradient_descent.append(p)
# Separate STDP layers and other types of layers into different optimizers for distinct training methods

optimizer_gd = Adam(params_gradient_descent, lr=lr)
optimizer_stdp = SGD(params_stdp, lr=lr, momentum=0.)
scheduler = StepLR(optimizer_stdp, step_size=25, gamma=0.5)
# Define optimizers for STDP layers and other layers

encoder = encoding.PoissonEncoder()
# Define Poisson encoder

outputs = {}

train_csv = '../plot/train.csv'
with open(train_csv, mode='w', newline="") as f:
    fieldnames = ['Epoch', "Train_loss", "Train_acc", "Test_acc"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    # Record training process

for epoch in range(50):

    start = time.time()
    net.train()
    train_loss = 0
    train_samples = 0

    train_acc = 0
    max_acc = 0

    for idx, (image, label) in enumerate(train_loader, start=1):
        optimizer_gd.zero_grad()
        optimizer_stdp.zero_grad()
        image = image.to(device)
        image = image.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        # Add a time dimension to the image

        label = label.to(device)

        encoded_image = encoder(image)
        y = net(encoded_image).mean(0)
        loss = F.cross_entropy(y, label)
        # Take the mean across the time dimension of the network output

        loss.backward()
        optimizer_stdp.zero_grad()
        # After backward propagation, all layer parameters will have gradients, including those optimized by STDP
        # So we need to clear STDP gradients

        for i in range(stdp_learners.__len__()):
            stdp_learners[i].step(on_grad=True)
            # Update STDP parameters

        optimizer_gd.step()
        optimizer_stdp.step()
        scheduler.step()
        # Update optimizer parameters

        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (y.argmax(1) == label).float().sum().item()

        functional.reset_net(net)
        for i in range(stdp_learners.__len__()):
            stdp_learners[i].reset()
        # Reset the entire network

    net.eval()
    test_samples = 0
    test_acc = 0
    with torch.no_grad():
        for idx, (image, label) in enumerate(test_loader, start=1):
            image = image.to(device)
            image = image.unsqueeze(0).repeat(T, 1, 1, 1, 1)
            label = label.to(device)

            encoded_image = encoder(image)
            y = net(encoded_image).mean(0)

            test_samples += label.numel()
            test_acc += (y.argmax(1) == label).float().sum().item()

            functional.reset_net(net)
            for i in range(stdp_learners.__len__()):
                stdp_learners[i].reset()

    train_loss /= train_samples
    train_acc /= train_samples
    test_acc /= test_samples
    torch.cuda.empty_cache()

    if test_acc > max_acc:
        max_acc = test_acc
        torch.save(net.state_dict(), "../model/best_model.pth")
        # Save the model with the highest accuracy

    with open(train_csv, mode='a', newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({"Epoch": epoch+1, 'Train_loss': train_loss, "Train_acc": train_acc, "Test_acc": test_acc})

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%, Test Accuracy: {test_acc * 100:.2f}%, Time: {time.time() - start:.2f}")