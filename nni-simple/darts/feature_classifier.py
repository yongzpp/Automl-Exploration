import torch.nn as nn
import darts.build_feature as build
from nni.nas.pytorch import mutables

class FeatureLayer(mutables.MutableScope):
    def __init__(self, key, prev_labels, out_filters, dropout_rate):
        super().__init__(key)
        self.out_filters = out_filters
        self.dropout_rate = dropout_rate
        self.mutable = mutables.LayerChoice([
            nn.Identity(),
            nn.ReLU(),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate),
            build.LinearRelu(self.out_filters),
            build.LinearTanh(self.out_filters),
            build.LinearNormal(self.out_filters)
        ])
        '''
        if len(prev_labels) > 0:
            self.skipconnect = mutables.InputChoice(choose_from=prev_labels, n_chosen=None)
        else:
            self.skipconnect = None
        '''
    def forward(self, prev_layers):
        out = self.mutable(prev_layers[-1])
        '''
        if self.skipconnect is not None:
            connection = self.skipconnect(prev_layers[:-1])
            if connection is not None:
                out += connection
        '''
        return out

class FeatureTrainer(nn.Module):
    def __init__(self, num_layers=8, in_filters=4096, out_filters=4096, num_classes=10, dropout_rate=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.out_filters = out_filters
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList()
        self.stem = nn.Linear(in_filters, self.out_filters)

        labels = []

        for layer_id in range(self.num_layers):
            labels.append("layer_{}".format(layer_id))
            self.layers.append(FeatureLayer(labels[-1], labels[:-1], self.out_filters, self.dropout_rate))

        self.dropout = nn.Dropout(self.dropout_rate)
        self.dense = nn.Linear(self.out_filters, self.num_classes)

    def forward(self, x):
        cur = x
        if x.size() != self.out_filters:
            cur = self.stem(x)
        layers = [cur]
        for layer_id in range(self.num_layers):
            cur = self.layers[layer_id](layers)
            layers.append(cur)
        cur = cur.view(cur.size(0), -1)
        logits = self.dense(cur)
        return logits
