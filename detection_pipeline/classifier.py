import albumentations as A
import numpy as np
import pandas as pd
import torch
import torchvision

from albumentations.pytorch import ToTensorV2
# from model_constructor_grouped import SimpleMobileNetV2
from PIL import Image
from time import time
from torch import nn


class SimpleMobileNetV2(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=True).features
        self.size = torchvision.models.mobilenet_v2().classifier[-1].weight.shape[1]
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.size, num_classes)

    def forward(self, x):
        x = self.model(x)  # from feature extractor
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        y = self.classifier(x)
        return y


class Model:
    def __init__(self, model, weights, labels, height, width):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # chose device
        assert labels.split('.')[-1] in ['csv', 'pkl'], 'Use *.pkl or *.csv file for labels'  # validate labels format
        self.labels = pd.read_csv(labels) if labels.split('.') == 'csv' else pd.read_pickle(labels)  # read labels
        self.labels = self.labels['label'].to_dict()
        self.weights = weights
        self.model = model(num_classes=len(self.labels.keys()), dropout=0.8)
        self.state_dict = torch.load(weights, map_location=torch.device(self.device))  # create state_dict
        self.model.load_state_dict(self.state_dict, strict=True)  # load state_dict
        self.model.eval()
        self.height = height
        self.width = width
        # self.dummy_input = torch.tensor([1, 3, self.height, self.width])

    def predict(self, x, decoded=True):
        x1 = self.transform(x).to(self.device)
        y = self.model(x1)
        if decoded:
            y = self.labels[y.argmax(axis=1).detach().cpu().numpy().item()]
        return y



    def print(self):
        print(self.model)

    def transform(self, x):
        if isinstance(x, np.ndarray):
            pass
        elif isinstance(x, str):
            x = np.asarray(Image.open(x))
        elif isinstance(x, Image.Image):
            x = np.asarray(x)
        t = A.Compose([
            A.Resize(self.height, self.width, interpolation=1, p=1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        return t(image=x)['image'].unsqueeze(0)


if __name__=='__main__':

    def test():
        model = Model(model=SimpleMobileNetV2,
                      weights='classifier.pt',
                      labels='/Users/yauhenikavaliou/PycharmProjects/leaf_desease/dataset/labels.pkl',
                      height=224,
                      width=224)
        # model = model.model  # activating the model
        x1 = '/Users/yauhenikavaliou/PycharmProjects/leaf_desease/dataset/images/IMG_0227_pred12.jpg'
        x2 = Image.open(x1)
        x3 = np.asarray(x2)

        try:
            print('\ntest 1')
            since = time()
            y = model.predict(x1)
            t = time() - since
            print(y)
            print(f'test 1 pass - {t:.4f}s.!')
        except Exception as e:
            print(type(e))  # the exception type
            print(e.args)  # arguments stored in .args
            print(e)

        try:
            print('\ntest 2')
            since = time()
            y = model.predict(x2)
            t = time() - since
            print(y)
            print(f'test 2 pass - {t:.4f}s.!')
        except Exception as e:
            print(type(e))
            print(e.args)
            print(e)
            print(e.with_traceback())

        try:
            print('\ntest 3')
            since = time()
            y = model.predict(x3)
            t = time() - since
            print(y)
            print(f'test 3 pass - {t:.4f}s.!')
        except Exception as e:
            print(type(e))
            print(e.args)
            print(e)


    test()
