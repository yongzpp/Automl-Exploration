import os
import yaml
import time
import argparse
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data._utils
import torch.backends.cudnn as cudnn

import nni
import search
import dataset
from darts.image_classifier import CNN
from darts.feature_classifier import FeatureTrainer
from darts.text_classifier import Model

try:
    import net
except:
    pass

net = None
preprocessor = None
best_acc = 0.0
start_epoch = 0
criterion = None
optimizer = None
trainloader = None
testloader = None
train_ls = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(os.getcwd()+'/ss.yml') as f:
    data = yaml.safe_load(f)

#Move a tensor, tuple, list, or dict onto device.
def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(to_device(t, device) for t in obj)
    if isinstance(obj, list):
        return [to_device(t, device) for t in obj]
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):
        return obj

#preprocess images
def preprocess(args):
    global trainloader
    global testloader

    print('==> Preparing data..')
    batch_size = data["batch_size"]
    if "batch_size" in args:
        batch_size = args["batch_size"]
    trainloader, testloader = dataset.getDataLoader(batch_size)

#preprocess features
def preprocess_features(args):
    global trainloader
    global testloader
    global train_ls

    batch_size = data["batch_size"]
    if "batch_size" in args:
        batch_size = args["batch_size"]
    trainloader, testloader = dataset.getDataLoader(batch_size)
    weights = torch.load("./features/ckpt.t7")
    train_ls = weights["train_ls"]

#preprocess texts
def preprocess_text(args):
    global trainloader
    global testloader
    global embedding

    print('==> Preparing data..')
    batch_size = data["batch_size"]
    if "batch_size" in args:
        batch_size = args["batch_size"]
    trainloader, testloader, embedding = dataset.getTextLoader(batch_size)

#prepare the environment and model for training according to search space
def prepare(args):
    global net
    global criterion
    global optimizer
    global train_ls
    global embedding

    NUM_CLASS = data["num_class"]
    if data["type"] == 'feature':
        net = search.get_feature(args, NUM_CLASS, train_ls)
    elif data["type"] == 'image':
        net = search.get_model(args, NUM_CLASS)
    elif data["type"] == 'text':
        net = search.get_text(args, NUM_CLASS, embedding)
    net = net.to(device)
    lr = 0.01
    momentum=0.9
    weight_decay=5e-4
    if (args["lr"]):
        lr = args["lr"]
    if (args["momentum"]):
        momentum = args["momentum"]
    criterion = nn.CrossEntropyLoss()
    optimizer = search.get_optimizer(args, lr, momentum, net)

#training of the model chosen
def train(epoch, batches=-1):
    global net
    global criterion
    global optimizer
    global trainloader

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = to_device(inputs, device), to_device(targets, device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
        if batches > 0 and (batch_idx+1) >= batches:
            return

#validation of the model
def test(epoch, args):
    global best_acc
    global testloader
    global net
    global criterion
    global optimizer

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    counter = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = to_device(inputs, device), to_device(targets, device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100.*correct/total
            counter += 1
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': args["model"],
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    return acc, best_acc

#set up environment and model for tabular data
def prepare_tabular(args):
    global net
    global preproccesor
    net = search.get_tabular(args)
    preproccesor = search.get_preprocessor(args)

#train and validate model for tabular data
def train_tabular(X_train, X_test, y_train, y_test):
    global net
    global preproccesor
    X_train = preproccesor.fit_transform(X_train)
    X_test = preproccesor.transform(X_test)
    net.fit(X_train, y_train)
    result = net.score(X_test, y_test)
    return result

#train and submit validate results to get next parameters
if __name__ == '__main__':
    try:
        RCV_CONFIG = nni.get_next_parameter()
        epochs = data["epochs"]
        if ("epochs" in RCV_CONFIG):
            epochs = RCV_CONFIG["epochs"]
        acc = 0.0
        best_acc = 0.0
        if (data["type"] == "tabular"):
            X_train, X_test, y_train, y_test = dataset.preprocess_tabular()
            prepare_tabular(RCV_CONFIG)
            best_acc = train_tabular(X_train, X_test, y_train, y_test)
            nni.report_final_result(best_acc)
        elif (data["type"] == "image"):
            preprocess(RCV_CONFIG)
            prepare(RCV_CONFIG)
            for epoch in range(start_epoch, start_epoch+epochs):
                train(epoch)
                score, best_score = test(epoch, RCV_CONFIG)
                nni.report_intermediate_result(score)
            nni.report_final_result(best_score)
        elif (data["type"] == "feature"):
            preprocess_features(RCV_CONFIG)
            prepare(RCV_CONFIG)
            for epoch in range(start_epoch, start_epoch+epochs):
                train(epoch)
                score, best_score = test(epoch, RCV_CONFIG)
                nni.report_intermediate_result(score)
            nni.report_final_result(best_score)
        elif (data["type"] == "text"):
            preprocess_text(RCV_CONFIG)
            prepare(RCV_CONFIG)
            for epoch in range(start_epoch, start_epoch+epochs):
                train(epoch)
                score, best_score = test(epoch, RCV_CONFIG)
                nni.report_intermediate_result(score)
            nni.report_final_result(best_score)

    except Exception as exception:
        raise
