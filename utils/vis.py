import matplotlib.pyplot as plt
import numpy as np

import pdb

def postprocess(img):
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)

    return img

"""
generate and display saliency maps
images - image tensors (B, C, H, W)
saliency_maps - the same shape as images
"""
def show_saliency_maps(images, saliency_maps):
    N = images.shape[0] # number of images

    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(postprocess(images[i].data.numpy().transpose(1, 2, 0)))
        plt.axis('off')
        plt.subplot(2, N, N + i + 1)
        plt.imshow(postprocess(saliency_maps[i].numpy()), cmap = plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()