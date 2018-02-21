import torch.nn as nn

from utils import *

class Model(nn.Module):
    def __init__(self, model_id):
        super(Model, self).__init__()

        self.model_id = model_id
        self.pretrained = load_model(model_id)

    def forward(self, x):
        scores = self.pretrained(x)

        return scores
