import torch
from torch.autograd import Variable
from torchvision import models, transforms

from PIL import Image
import numpy as np

import os
import sys

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
def load_image(filename):
    img = Image.open(filename)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    return Variable(preprocess(img).unsqueeze(0))   # (B, C, H, W) = (1, 3, H, W)

# pick five images from a given directory randomly; dataset is from cs231n Assignment3
def pick_images(dir = '/localdisk/dataset/tiny-imagenet-100-A/val/images'):
    filename_list = np.random.choice(os.listdir(dir), size = 5)
    filename_list = [os.path.join(dir, x) for x in filename_list]

    return filename_list

# generate input image variables
def gen_images(filename_list):
    assert(len(filename_list) == 5)

    images = [load_image(x) for x in filename_list]
    images = torch.cat(images, dim = 0)  # (B, C, H, W) = (5, 3, 224, 224)

    return images

# get ground-truth labels for five picked images
def get_labels(filename_list):
    # load 1000 categories ids
    clsloc_file = '/localdisk/dataset/tiny-imagenet-100-A/clsloc.txt'
    with open(clsloc_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    clslabels = dict()
    for line in content:    # id, label, description
        line = line.split(' ')
        line = [x.strip() for x in line]
        clslabels[line[0]] = line[1]

    gt_file = '/localdisk/dataset/tiny-imagenet-100-A/val/val_annotations.txt'
    with open(gt_file) as f:
        gt = f.readlines()
    gt = [x.strip() for x in gt]

    gt_dict = dict()
    for line in gt: # filename, id, cls1, cls2, cls3, cls4
        line = line.split('\t')
        line = [x.strip() for x in line]
        gt_dict[line[0]] = line[1]

    labels = []
    for filename in filename_list:
        filename = filename[filename.rfind('/') + 1:]
        if filename in gt_dict:
            id = gt_dict[filename]
            labels.append(clslabels[id])

    # make sure get all five labels
    assert(len(labels) == 5)

    return labels