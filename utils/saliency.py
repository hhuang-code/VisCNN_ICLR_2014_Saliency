import torch
from torch.autograd import Variable

import pdb

"""
Use image(x) and label(y) to compute correct saliency map
Args:
    x - input images Varialbe: (B, C, H, W) = (B, 3, H, W)
    y - labels, should be a LongTensor: (N,), N is the number of input images
    model - a pretrained model
Return:
    saliency maps, a tensor of shape (B, H, W), one per input image
"""
def compute_saliency_maps(x, y, model):
    if not isinstance(x, Variable):
        x = Variable(x, requires_grad = True)
    elif not x.requires_grad:
        x.requires_grad = True

    if not isinstance(y, Variable):
        y = Variable(y)

    # forward pass
    scores = model(x)

    scores = scores.gather(1, y.view(-1, 1)).squeeze()

    # backward pass
    scores.backward(torch.ones(scores.shape))

    saliency_maps = x.grad.data

    saliency_maps = saliency_maps.abs()
    saliency_maps, idx = torch.max(saliency_maps, dim = 1)    # get max abs from all (3) channels

    return saliency_maps
