import os
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import logging
import pickle
import search
from collections import Counter

with open(os.getcwd()+'/ss.yml') as f:
    data = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = logging.getLogger("nas")

'''
IMAGE DATA
'''
#image dataset
class Data(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.annotations = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        image = Image.fromarray(image)

        y_label = int(self.annotations.iloc[index, 1])
        y_label = torch.tensor(y_label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)
        return (image, y_label)

#preprocess image directory to csv file for creating dataset
def folder_structure_to_csv(path:str, save_path:str):
    counter = 0
    paths = []
    labels = []
    for folder in os.listdir(path):
        for image in os.listdir(path+'/'+folder):
            paths.append(folder+'/'+image)
            labels.append(counter)
        counter += 1
    df = pd.DataFrame({'image_paths': paths,'label': labels})
    df.to_csv(save_path, index=None, encoding=None)

#preprocessing image dataset
def preprocess_img():
    train_ls = []
    test_ls = []
    for i in data["transform"]:
        if (i["_type"] == "train"):
            if (i["_name"] == "resize"):
                train_ls.append(transforms.Resize((i["_values"][0], i["_values"][1])))
            if (i["_name"] == "center_crop"):
                train_ls.append(transforms.CenterCrop((i["_values"][0], i["_values"][1])))
            if (i["_name"] == "pad"):
                train_ls.append(transforms.Pad((i["_values"][0], i["_values"][1])))
            if (i["_name"] == "random_crop"):
                train_ls.append(transforms.RandomCrop((i["_values"][0], i["_values"][1])))
            if (i["_name"] == "random_horizontal_flip"):
                train_ls.append(transforms.RandomHorizontalFlip(i["_values"][0]))
            if (i["_name"] == "random_rotation"):
                train_ls.append(transforms.RandomRotation(i["_values"][0]))
            if (i["_name"] == "random_vertical_flip"):
                train_ls.append(transforms.RandomVerticalFlip(i["_values"][0]))
        else:
            if (i["_name"] == "resize"):
                test_ls.append(transforms.Resize((i["_values"][0], i["_values"][1])))
            if (i["_name"] == "center_crop"):
                train_ls.append(transforms.CenterCrop((i["_values"][0], i["_values"][1])))
            if (i["_name"] == "pad"):
                train_ls.append(transforms.Pad((i["_values"][0], i["_values"][1])))
            if (i["_name"] == "random_crop"):
                train_ls.append(transforms.RandomCrop((i["_values"][0], i["_values"][1])))
            if (i["_name"] == "random_horizontal_flip"):
                train_ls.append(transforms.RandomHorizontalFlip(i["_values"][0]))
            if (i["_name"] == "random_rotation"):
                train_ls.append(transforms.RandomRotation(i["_values"][0]))
            if (i["_name"] == "random_vertical_flip"):
                train_ls.append(transforms.RandomVerticalFlip(i["_values"][0]))
    train_ls.append(transforms.ToTensor())
    test_ls.append(transforms.ToTensor())

    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    for i in data["transform"]:
        if (i["_type"] == "train"):
            if (i["_name"] == "normalize" and i["_values"] == "True"):
                train_ls.append(transforms.Normalize(MEAN, STD))
        else:
            if (i["_name"] == "normalize" and i["_values"] == "False"):
                train_ls.append(transforms.Normalize(MEAN, STD))

    transform_train = transforms.Compose(train_ls)
    transform_test = transforms.Compose(test_ls)
    trainset = Data(csv_file=data["train_dir"]+"/train.csv",
                                root_dir=data["train_dir"],
                                transform=transform_train)
    testset = Data(csv_file=data["val_dir"]+"/val.csv",
                                root_dir=data["val_dir"],
                                transform=transform_test)
    return trainset, testset

#get image dataloader
def getImageLoader(batch_size=32):
    if (not os.path.isfile(data["train_dir"]+"/train.csv")):
        folder_structure_to_csv(data["train_dir"], data["train_dir"]+"/train.csv")
    if (not os.path.isfile(data["val_dir"]+"/val.csv")):
        folder_structure_to_csv(data["val_dir"], data["val_dir"]+"/val.csv")
    trainset, testset = preprocess_img()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                    shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                    shuffle=False, num_workers=1)
    return trainloader, testloader

'''
FEATURES
'''
#feature dataset
class Features(Dataset):
    def __init__(self, features, transform=None):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (self.features[index][0], self.features[index][1])

#extract features based on the preferred extractor (image)
def extract_features():
    trainloader, testloader = getImageLoader(1)
    net = search.get_extractor(data["feature"]["extract"])
    net = net.to(device)
    net.eval()
    train_ls = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
          inputs, targets = inputs.to(device), targets.to(device)
          outputs = net(inputs)
          outputs = outputs.view(1,-1)
          train_ls.append((outputs.to('cpu'), targets.to('cpu')[0]))
    test_ls = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
          inputs, targets = inputs.to(device), targets.to(device)
          outputs = net(inputs)
          outputs = outputs.view(1,-1)
          test_ls.append((outputs.to('cpu'), targets.to('cpu')[0]))

    state = {
        'train_ls': train_ls,
        'test_ls': test_ls,
    }
    if not os.path.isdir('features'):
        os.mkdir('features')
    torch.save(state, './features/ckpt.t7')

#preprocess features data
def preprocess_features():
    if (not os.path.isfile('./features/ckpt.t7')):
        extract_features()
    weights = torch.load("./features/ckpt.t7")
    train_ls = weights["train_ls"]
    test_ls = weights["test_ls"]
    trainset = Features(features = train_ls)
    testset = Features(features = test_ls)
    return trainset, testset

#get the dataloader according to the data type
def getDataLoader(batch_size=32):
    if (data["type"] == "image"):
        if (not os.path.isfile(data["train_dir"]+"/train.csv")):
            folder_structure_to_csv(data["train_dir"], data["train_dir"]+"/train.csv")
        if (not os.path.isfile(data["val_dir"]+"/val.csv")):
            folder_structure_to_csv(data["val_dir"], data["val_dir"]+"/val.csv")
        trainset, testset = preprocess_img()
    if (data["type"] == "text"):
        trainset, testset, embedding = preprocess_text()
    if (data["type"] == "feature"):
        trainset, testset = preprocess_features()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                    shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                    shuffle=False, num_workers=1)
    return trainloader, testloader

'''
TABULAR DATA
'''
#preprocess tabular data
def preprocess_tabular():
    '''
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, random_state=99, test_size=0.25)
    '''
    df = pd.read_csv(data["train_dir"])
    label = data["label"]
    y = df.pop(label).to_frame()
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    return X_train, X_test, y_train, y_test

'''
TEXT DATA
'''
#text dataset
class Text(Dataset):
    def __init__(self, sents, mask, labels):
        self.sents = sents
        self.labels = labels
        self.mask = mask

    def __getitem__(self, index):
        return (self.sents[index], self.mask[index]), self.labels[index]

    def __len__(self):
        return len(self.sents)

#get text dataloader
def getTextLoader(batch_size=32):
    trainset, testset, embedding = preprocess_text()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                    shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                    shuffle=False, num_workers=1)
    return trainloader, testloader, embedding

#load pretrained embeddings
def load_glove_model(filename, embed_dim):
    if os.path.exists(filename + ".cache"):
        logger.info("Found cache. Loading...")
        with open(filename + ".cache", "rb") as fp:
            return pickle.load(fp)
    embedding = {"mapping": dict(), "pool": []}
    with open(filename) as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            vocab_word, *vec = line.rsplit(" ", maxsplit=embed_dim)
            assert len(vec) == 300, "Unexpected line: '%s'" % line
            embedding["pool"].append(np.array(list(map(float, vec)), dtype=np.float32))
            embedding["mapping"][vocab_word] = i
    embedding["pool"] = np.stack(embedding["pool"])
    with open(filename + ".cache", "wb") as fp:
        pickle.dump(embedding, fp)
    return embedding

def init_trainable_embedding(embedding_path, word_id_dict, embed_dim=300):
    word_embed_model = load_glove_model(embedding_path, embed_dim)
    assert word_embed_model["pool"].shape[1] == embed_dim
    embedding = np.random.random([len(word_id_dict), embed_dim]).astype(np.float32) / 2.0 - 0.25
    embedding[0] = np.zeros(embed_dim)  # PAD
    embedding[1] = (np.random.rand(embed_dim) - 0.5) / 2  # UNK
    for word in sorted(word_id_dict.keys()):
        idx = word_id_dict[word]
        if idx == 0 or idx == 1:
            continue
        if word in word_embed_model["mapping"]:
            embedding[idx] = word_embed_model["pool"][word_embed_model["mapping"][word]]
        else:
            embedding[idx] = np.random.rand(embed_dim) / 2.0 - 0.25
    return embedding

def get_id_input(content, word_id_dict, max_input_length):
    words = content.split(" ")
    sentence = [word_id_dict["<pad>"]] * max_input_length
    mask = [0] * max_input_length
    unknown = word_id_dict["<unknown>"]
    for i, word in enumerate(words[:max_input_length]):
        sentence[i] = word_id_dict.get(word, unknown)
        mask[i] = 1
    return sentence, mask

def get_word_id_dict(word_num_dict, word_id_dict, min_count):
    z = [k for k in sorted(word_num_dict.keys())]
    for word in z:
        count = word_num_dict[word]
        if count >= min_count:
            index = len(word_id_dict)
            if word not in word_id_dict:
                word_id_dict[word] = index
    return word_id_dict

def load_word_num_dict(phrases, word_num_dict):
    for sentence, _ in phrases:
        words = sentence.split(" ")
        for cur_word in words:
            word = cur_word.strip()
            word_num_dict[word] += 1
    return word_num_dict

def get_trainable_data(phrases, word_id_dict, max_input_length):
    texts, labels, mask = [], [], []
    for phrase, label in phrases:
        if not phrase.split():
            continue
        phrase_split, mask_split = get_id_input(phrase, word_id_dict, max_input_length)
        texts.append(phrase_split)
        labels.append(int(label))
        mask.append(mask_split)  # field_input is mask
    labels = np.array(labels, dtype=np.int64)
    texts = np.reshape(texts, [-1, max_input_length]).astype(np.int32)
    mask = np.reshape(mask, [-1, max_input_length]).astype(np.int32)
    return Text(texts, mask, labels)

def get_phrases(path):
    with open(path) as fp:
        all_phrases = []
        while True:
            line = fp.readline()
            ls = line.split("|")
            label = None
            stripped = None
            for i in range(len(ls)):
                if i == 1:
                    label = int(ls[i].strip())
                else:
                    stripped = ls[i].strip()
            temp = (stripped, label)
            all_phrases.append(temp)
            if line == "":
                break
    return all_phrases

#preprocess text data
def preprocess_text(max_input_length=64, min_count=1, train_with_valid=False,
                  train_ratio=1., valid_ratio=1., is_binary=False, only_sentence=False):
    word_id_dict = dict()
    word_num_dict = Counter()
    train_file_name = data["train_dir"]
    valid_file_name = data["val_dir"]
    logger.info("Reading data...")
    train_phrases = get_phrases(train_file_name)
    logger.info("Finish load train phrases.")
    valid_phrases = get_phrases(valid_file_name)
    logger.info("Finish load valid phrases.")

    word_id_dict["<pad>"] = 0
    word_id_dict["<unknown>"] = 1
    load_word_num_dict(train_phrases, word_num_dict)
    logger.info("Finish load train words: %d.", len(word_num_dict))
    load_word_num_dict(valid_phrases, word_num_dict)
    logger.info("Finish load valid words: %d.", len(word_num_dict))
    word_id_dict = get_word_id_dict(word_num_dict, word_id_dict, min_count)
    logger.info("After trim vocab length: %d.", len(word_id_dict))

    logger.info("Loading embedding...")
    embedding = init_trainable_embedding(os.path.join("./embedding", "glove.6B.300d.txt"), word_id_dict)
    logger.info("Finish initialize word embedding.")

    dataset_train = get_trainable_data(train_phrases, word_id_dict, max_input_length)
    logger.info("Loaded %d training samples.", len(dataset_train))
    dataset_valid = get_trainable_data(valid_phrases, word_id_dict, max_input_length)
    logger.info("Loaded %d validation samples.", len(dataset_valid))
    return dataset_train, dataset_valid, torch.from_numpy(embedding)
