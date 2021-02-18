import os
import yaml
import time
import logging
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim

import nni
from nni.nas.pytorch import darts
from nni.nas.pytorch.spos import SPOSSupernetTrainingMutator, SPOSSupernetTrainer
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback, ModelCheckpoint

from darts.image_classifier import CNN
from darts.text_classifier import Model
from darts.feature_classifier import FeatureTrainer
from darts.utils import accuracy

import search
import dataset
from dataset import Data, preprocess_img, preprocess_features, preprocess_text

with open(os.getcwd()+'/ss.yml') as f:
    data = yaml.safe_load(f)

if __name__ == "__main__":
    parser = ArgumentParser("nas")
    parser.add_argument("--lr", default=0.025, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--layers", default=4, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--channels", default=16, type=int)
    parser.add_argument("--dropout_rate", default=0.0, type=float)
    parser.add_argument("--mutator_lr", default=3.0E-4, type=float)
    parser.add_argument("--mutator_momentum", default=1.0E-3, type=float)
    parser.add_argument("--unrolled", default=False, action="store_true")
    parser.add_argument("--visualization", default=False, action="store_true")
    args = parser.parse_args()

    logger = logging.getLogger('nas')

    NUM_CLASS = data["num_class"]
    criterion = nn.CrossEntropyLoss()
    params = nni.get_next_parameter()

    #accomodate search space from user
    if "epochs" in data:
        args.epochs = data["epochs"]
    if "epochs" in params:
        args.epochs = params["epochs"]
    if "batch_size" in data:
        args.batch_size = data["batch_size"]
    if "batch_size" in params:
        args.batch_size = params["batch_size"]
    if ("lr" in params):
        args.lr = params["lr"]
    if ("momentum" in params):
        args.momentum = params["momentum"]
    if ("mutator_lr" in params):
        args.mutator_lr = params["mutator_lr"]
    if ("mutator_momentum" in params):
        arg.mutator_momentum = params["mutator_momentum"]
    if ("channels" in params):
        args.channels = params["channels"]
    if ("channels" in data["nas"]):
        channels = data["nas"]["channels"]
    if ("layers" in params):
        args.layers = params["layers"]
    if ("layers" in data["nas"]):
        layers = data["nas"]["layers"]
    if ("dropout_rate" in data["nas"]):
        args.dropout_rate = data["nas"]["dropout_rate"]

    #preprocessing for different types of input
    if (data["type"] == "feature"):
        dataset_train, dataset_valid = preprocess_features()
        weights = torch.load("./features/ckpt.t7")
        train_ls = weights["train_ls"]
        model = FeatureTrainer(num_layers = args.layers,
                               in_filters = list(train_ls[0][0].size())[1],
                               out_filters = args.channels,
                               dropout_rate = args.dropout_rate,
                               num_classes = NUM_CLASS)
    if (data["type"] == "text"):
        dataset_train, dataset_valid, embedding = preprocess_text()
        model = Model(embedding)
    if (data["type"] == "image"):
        if (not os.path.isfile(data["train_dir"]+"/train.csv")):
            dataset.folder_structure_to_csv(data["train_dir"], data["train_dir"]+"/train.csv")
        if (not os.path.isfile(data["val_dir"]+"/val.csv")):
            dataset.folder_structure_to_csv(data["val_dir"], data["val_dir"]+"/val.csv")
        dataset_train, dataset_valid = preprocess_img()
        model = CNN(input_size = data["nas"]["input_size"],
                    channels = args.channels,
                    in_channels = data["nas"]["in_channels"],
                    n_layers = args.layers,
                    n_classes = NUM_CLASS)

    optim = search.get_optimizer(params, args.lr, args.momentum, model)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)

    #different trainers according to the user
    if data["nas"]["type"] == "darts":
        trainer = darts.DartsTrainer(model,
                                   loss=criterion,
                                   metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                                   optimizer=optim,
                                   num_epochs=args.epochs,
                                   dataset_train=dataset_train,
                                   dataset_valid=dataset_valid,
                                   batch_size=args.batch_size,
                                   log_frequency=args.log_frequency,
                                   callbacks=[LRSchedulerCallback(lr_scheduler), ModelCheckpoint("./weights"),
                                        ArchitectureCheckpoint("./architectures")])
        mutator = trainer.mutator
        trainer.ctrl_optim = search.get_mutator_optimizer(params, args.mutator_lr, args.mutator_momentum, mutator)

    if data["nas"]["type"] == "spos":
        train_loader, val_loader = dataset.getDataLoader(args.batch_size)
        trainer = SPOSSupernetTrainer(model,
                                   loss=criterion,
                                   metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                                   optimizer=optim,
                                   num_epochs=args.epochs,
                                   train_loader = train_loader,
                                   valid_loader = val_loader,
                                   batch_size=args.batch_size,
                                   log_frequency=args.log_frequency,
                                   callbacks=[LRSchedulerCallback(lr_scheduler), ModelCheckpoint("./weights"),
                                        ArchitectureCheckpoint("./architectures")])
    trainer.train()
