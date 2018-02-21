from utils import *
from models import *

import pdb

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Wrong number of arguments!')

    try:
        model_id = int(sys.argv[1])
    except:
        sys.exit('Wrong second argument')

    filename_list = pick_images()   # randomly pick five images
    images = gen_images(filename_list)  # generate input images (Variable)
    labels = get_labels(filename_list)  # get ground-truth labels
    print('labels: ', labels)

    model = Model(model_id)    # model_id: 1-AlexNet, 2-VGG16, 3-ResNet101
    model.eval()

    scores= model(images)  # (5, 1000)

    max_scores, preds = scores.max(dim = 1)


    print('max_prob: ', max_scores)
    print('preds: ', preds)

    # turn labels into a tensor
    labels = [int(x) for x in labels]
    labels = np.array(labels)
    labels = torch.LongTensor(labels)

    # compute saliency maps using gradient of scores
    saliency_maps = compute_saliency_maps(images, labels, model)
    # display saliency maps
    show_saliency_maps(images, saliency_maps)

    # compute saliency maps using cross entropy loss
    saliency_maps = compute_saliency_maps_crossentropy(images, labels, model)
    # display saliency maps
    show_saliency_maps(images, saliency_maps)

