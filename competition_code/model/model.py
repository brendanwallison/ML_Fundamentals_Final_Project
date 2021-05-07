import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import resnest.torch as resnest_torch

# from https://www.kaggle.com/kneroma/clean-fast-simple-bird-identifier-training-colab
class PretrainedModel():
    def __init__(self, num_classes=397, name="resnest"):
        """
        Loads a pretrained model. 
        Supports ResNest, ResNext-wsl, EfficientNet, ResNext and ResNet.

        Arguments:
            name {str} -- Name of the model to load

        Keyword Arguments:
            num_classes {int} -- Number of classes to use (default: {1})

        Returns:
            torch model -- Pretrained model
        """
        if "resnest" in name:
            model = getattr(resnest_torch, name)(pretrained=True)
        elif "wsl" in name:
            model = torch.hub.load("facebookresearch/WSL-Images", name)
        elif name.startswith("resnext") or  name.startswith("resnet"):
            model = torch.hub.load("pytorch/vision:v0.6.0", name, pretrained=True)
        elif name.startswith("tf_efficientnet_b"):
            model = getattr(timm.models.efficientnet, name)(pretrained=True)
        elif "efficientnet-b" in name:
            model = EfficientNet.from_pretrained(name)
        else:
            model = pretrainedmodels.__dict__[name](pretrained='imagenet')

        if hasattr(model, "fc"):
            nb_ft = model.fc.in_features
            model.fc = nn.Linear(nb_ft, num_classes)
        elif hasattr(model, "_fc"):
            nb_ft = model._fc.in_features
            model._fc = nn.Linear(nb_ft, num_classes)
        elif hasattr(model, "classifier"):
            nb_ft = model.classifier.in_features
            model.classifier = nn.Linear(nb_ft, num_classes)
        elif hasattr(model, "last_linear"):
            nb_ft = model.last_linear.in_features
            model.last_linear = nn.Linear(nb_ft, num_classes)
        self.model = model

    def get_model(self):
        return self.model
    


class BirdsongModel2(BaseModel):
    def __init__(self, num_classes=397):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1) # Input dimensions = output dimensions
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, padding=1) # Input dimensions = output dimensions
        self.conv3 = nn.Conv2d(12, 24, kernel_size=3, padding=1) # Input dimensions = output dimensions
        self.conv4 = nn.Conv2d(24, 48, kernel_size=3, padding=1) # Input dimensions = output dimensions
        self.fc1 = nn.Linear(48 * 8 * 17, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, 2)) # max pool cuts each dimension in half
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2)) # max pool cuts each dimension in half
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2)) # max pool cuts each dimension in half
        x = F.relu(F.max_pool2d(self.conv4(x), 2, 2)) # max pool cuts each dimension in half
        x = x.view(-1, 48 * 8 * 17)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sig(x)


class BirdsongModel(BaseModel):
    def __init__(self, num_classes=398):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1) # Input dimensions = output dimensions
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1) # Input dimensions = output dimensions
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # Input dimensions = output dimensions
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # Input dimensions = output dimensions
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Input dimensions = output dimensions
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.m = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, 2)) # max pool cuts each dimension in half
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2)) # max pool cuts each dimension in half
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2)) # max pool cuts each dimension in half
        x = F.relu(F.max_pool2d(self.conv4(x), 2, 2)) # max pool cuts each dimension in half
        x = F.relu(F.max_pool2d(self.conv5(x), 2, 2)) # max pool cuts each dimension in half
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.m(x)



class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)