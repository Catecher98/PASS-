import torch.nn as nn
import torch

class network(nn.Module):
    def __init__(self, numclass, feature_extractor, task_size=0):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.mid = None
        self.fc = nn.Linear(512, numclass, bias=True)
        self.task_size = task_size

    def forward(self, input):
        x, feature_mid = self.feature(input)
        self.mid = feature_mid
        x = self.fc(x)
        return x

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight[:out_feature]
        self.fc.bias.data[:out_feature] = bias[:out_feature]

    def feature_extractor(self, inputs):
        return self.feature(inputs)

    def weight_aligning(self):
      pass
