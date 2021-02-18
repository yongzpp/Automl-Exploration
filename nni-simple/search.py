import torch.backends.cudnn as cudnn

import torchvision
import torchvision.models as models

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer

import nni
from nni.nas.pytorch.fixed import apply_fixed_architecture

import dataset
from darts.image_classifier import CNN
from darts.feature_classifier import FeatureTrainer
from darts.text_classifier import Model

try:
    import net
except:
    pass

with open(os.getcwd()+'/ss.yml') as f:
    data = yaml.safe_load(f)

#get preprocessor for tabular data (can be extended)
def get_preprocessor(args):
    if ("preprocessor" in args):
        if args['preprocessor'] == 'standard':
            return StandardScaler()
        if args['preprocessor'] == 'robust':
            return RobustScaler()
        if args['preprocessor'] == 'minmax':
            return MinMaxScaler()
        if args['preprocessor'] == 'normalize':
            return Normalizer()
    return StandardScaler()

#get optimizer for all data (can be extended)
def get_optimizer(args, lr, momentum, net):
    if ("optimizer" in args):
        if args['optimizer'] == 'SGD':
            return optim.SGD(net.parameters(), lr, momentum, weight_decay=5e-4)
        if args['optimizer'] == 'Adadelta':
            return optim.Adadelta(net.parameters(), lr)
        if args['optimizer'] == 'Adagrad':
            return optim.Adagrad(net.parameters(), lr)
        if args['optimizer'] == 'Adam':
            return optim.Adam(net.parameters(), lr)
        if args['optimizer'] == 'Adamax':
            return optim.Adamax(net.parameters(), lr)
    return optim.SGD(net.parameters(), lr, momentum, weight_decay=5e-4)

#get optimizer for architecture trainer/ darts (can be extended)
def get_mutator_optimizer(args, lr, momentum, mutator):
    if ("mutator_optimizer" in args):
        if args['mutator_optimizer'] == 'SGD':
            return torch.optim.SGD(mutator.parameters(), lr, momentum, weight_decay=1.0E-3)
        if args['mutator_optimizer'] == 'Adadelta':
            return torch.optim.Adadelta(mutator.parameters(), lr)
        if args['mutator_optimizer'] == 'Adagrad':
            return torch.optim.Adagrad(mutator.parameters(), lr)
        if args['mutator_optimizer'] == 'Adam':
            return torch.optim.Adam(mutator.parameters(), lr)
        if args['mutator_optimizer'] == 'Adamax':
            return torch.optim.Adamax(mutator.parameters(), lr)
    return torch.optim.Adam(mutator.parameters(), lr, betas=(0.5, 0.999), weight_decay=1.0E-3)

#get extractor to extract features from data
def get_extractor(model):
    if model == 'vgg':
        net = models.vgg.vgg19(pretrained = True)
        for param in net.parameters():
            param.requires_grad = False
        num_features = net.classifier[6].in_features
        features = list(net.classifier.children())[:-1]
        net.classifier = nn.Sequential(*features)
    if model == 'resnet':
        net = models.resnet50(pretrained=True)
        for p in net.parameters():
            p.requires_grad = False
        modules=list(net.children())[:-1]
        net=nn.Sequential(*modules)
    return net

#get the models from tabular data (can be extended)
def get_tabular(args):
    if args['model']["_name"] == 'logreg':
        net = LogisticRegression()
        if ("C" in args["model"]):
            net.C = args["model"]["C"]
        if ("penalty" in args["model"]):
            net.penalty = args["model"]["penalty"]
    if args['model']["_name"] == 'svc':
        net = SVC()
        if ("C" in args["model"]):
            net.C = args["model"]["C"]
        if ("kernel" in args["model"]):
            net.kernel = args["model"]["kernel"]
    if args['model']["_name"] == 'rf':
        net = RandomForestClassifier()
        if ("n_estimators" in args["model"]):
            net.n_estimators = args["model"]["n_estimators"]
        if ("criterion" in args["model"]):
            net.criterion = args["model"]["criterion"]
        if ("max_depth" in args["model"]):
            net.max_depth = args["model"]["max_depth"]
        if ("max_leaf_nodes" in args["model"]):
            net.max_leaf_nodes = args["model"]["max_leaf_nodes"]
        if ("min_samples_leaf" in args["model"]):
            net.min_samples_leaf = args["model"]["min_samples_leaf"]
    if args['model']["_name"] == 'knn':
        net = KNeighborsClassifier()
        if ("n_neighbors" in args["model"]):
            net.n_neighbors = args["model"]["n_neighbors"]
        if ("weights" in args["model"]):
            net.weights = args["model"]["weights"]
    if args['model']["_name"] == 'gaussiannb':
        net = GaussianNB()
    if args['model']["_name"] == 'custom':
        net = Net()
    return net

#get the models from features (can be extended)
def get_feature(args, NUM_CLASS, features):
    if args['model'] == 'custom':
        net = Net()
    if args['model'] == 'nas':
        net = FeatureTrainer(num_layers = data["nas"]["layers"],
                             in_filters = list(features[0][0].size())[1],
                             out_filters = data["nas"]["channels"],
                             dropout_rate = data["nas"]["dropout_rate"],
                             num_classes = NUM_CLASS)
        if (os.path.isfile(data["nas"]["save_path"])):
            apply_fixed_architecture(net, data["nas"]["save_path"])
    return net

#get the models from text data (can be extended)
def get_text(args, NUM_CLASS, embedding):
    if args['model'] == 'custom':
        net = Net()
    if args['model'] == 'nas':
        net = Model(embedding)
        if (os.path.isfile(data["nas"]["save_path"])):
            apply_fixed_architecture(net, data["nas"]["save_path"])
    return net

#get the models from images data (can be extended)
def get_model(args, NUM_CLASS, features=None):
    if args['model'] == 'vgg':
        net = models.vgg.vgg19(pretrained = True)
        for param in net.parameters():
            param.requires_grad = False
        num_features = net.classifier[6].in_features
        features = list(net.classifier.children())[:-1]
        features.extend([nn.Linear(num_features, NUM_CLASS)])
        net.classifier = nn.Sequential(*features)
    if args['model'] == 'resnet':
        net = models.resnet.resnet50(pretrained = True)
        for param in net.parameters():
            param.requires_grad = False
        num_ftrs = net.fc.in_features
        net.fc =  nn.Linear(num_ftrs, NUM_CLASS)
    if args['model'] == 'mobilenet':
        net = models.mobilenet.mobilenet_v2(pretrained = True)
        for param in net.parameters():
            param.requires_grad = False
        net.classifier[1] = nn.Linear(net.last_channel, NUM_CLASS)
    if args['model'] == 'custom':
        net = Net()
    if args['model'] == 'nas':
        net = CNN(input_size = data["nas"]["input_size"],
                  channels = data["nas"]["channels"],
                  in_channels = data["nas"]["in_channels"],
                  n_layers = data["nas"]["layers"],
                  n_classes = NUM_CLASS)
        if (os.path.isfile(data["nas"]["save_path"])):
            apply_fixed_architecture(net, data["nas"]["save_path"])
    return net
