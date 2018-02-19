import torch
from torch.autograd import Variable
from torchvision import models, transforms

from PIL import Image
import numpy as np

import sys

from utils import *

import pdb

# load a pretrained model
# model_id: 1-AlexNet, 2-VGG16, 3-ResNet101
def load_model(model_id = 2):
    if model_id == 1:
        model = models.alexnet(pretrained = True)
    elif model_id == 2:
        model = models.vgg16(pretrained = True)
    elif model_id == 3:
        model = models.resnet101(pretrained = True)
    else:
        sys.exit('No such model!')

    return model

# load an image
def load_image(filename = './lena.jpg'):
    img = Image.open(filename)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    return Variable(preprocess(img).unsqueeze(0))   # (B, C, H, W) = (1, 3, H, W)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Wrong number of arguments!')

    try:
        model_id = int(sys.argv[1])
    except:
        sys.exit('Wrong second argument')

    model = load_model(model_id)
    model.eval()

    filename1 = './lena.jpg'    # first image
    image1 = load_image(filename1)

    filename2 = './cat.jpg'     # second image
    image2 = load_image(filename2)

    images = torch.cat((image1, image2), dim = 0)   # (B, C, H, W) = (2, 3, 224, 224)

    scores = model(images)  # (2, 1000)

    values, labels = scores.max(dim = 1)

    print(values)
    print(labels)

    # compute saliency maps
    saliency_maps = compute_saliency_maps(images, labels, model)

    del model
    # display saliency maps
    show_saliency_maps(images, saliency_maps)

