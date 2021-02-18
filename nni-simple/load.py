#example of retraining and recreating
import os
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from darts.image_classifier import CNN
from darts.feature_classifier import FeatureTrainer
from darts.text_classifier import Model
from nni.nas.pytorch.fixed import apply_fixed_architecture

import dataset

with open(os.getcwd()+'/ss.yml') as f:
    data = yaml.safe_load(f)

NUM_CLASS = data["num_class"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#For NAS models
model = CNN(input_size = data["nas"]["input_size"],
            channels = 8,
            in_channels = data["nas"]["in_channels"],
            n_layers = 4,
            n_classes = NUM_CLASS)

#If weights from architecture training is preferred
checkpoint = torch.load("/home/yziping/data2/demo/weights/trial3/epoch_41.pth.tar")
model.load_state_dict(checkpoint)

#If weights from best run in HPO is preferred
checkpoint = torch.load("/home/yziping/data/nni_modified/checkpoint/ckpt.t7")
net.load_state_dict(checkpoint["net"])

#Applicable only for NAS
apply_fixed_architecture(model, "/home/yziping/data2/final/architectures/trial4/epoch_19.json")

trainloader, testloader = dataset.getDataLoader(32)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

lr = 0.025
momentum=0.9
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr, momentum, weight_decay=3e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 150, eta_min=0.001)

def train(epoch, batches=-1):
    global model
    global criterion
    global optimizer
    global trainloader

    model.train()
    model.to(device)
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total

def test(epoch):
    global model
    global criterion
    global optimizer
    global testloader

    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    total = 0
    counter = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100.*correct/total
            counter += 1
    acc = 100.*correct/total
    return acc

if __name__ == '__main__':
    patience = 100
    counter = 0
    best_score = 0
    for epoch in range(150):
        train(epoch)
        score = test(epoch)
        if score < best_score:
            counter += 1
            if counter == patience:
                break
        else:
            best_score = score
            counter = 0
        print("Epoch {}: {}".format(epoch, score))
    print(best_score)
