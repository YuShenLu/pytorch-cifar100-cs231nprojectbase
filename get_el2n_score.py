from el2n import compute_el2n_score
import argparse
import numpy as np
import torch
import torchvision.models as models
import os
from conf import settings
from utils import get_network, get_training_dataloader, most_recent_folder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-path', type=str, required=True, help='path for model')
    args = parser.parse_args()

    #load model
    net = get_network(args)
    map_loc = torch.device('cpu')
    if args.gpu:
        map_loc = torch.device('cuda')
    net.load_state_dict(torch.load(args.path, map_location=map_loc))
    # net = models.resnet18(pretrained=True)

    # load training data
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=False
    )

    # compute el2n score
    scores = []
    num_batches = len(cifar100_training_loader.dataset) // args.b
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        print(f'computing el2n score batch {batch_index} of {num_batches}')

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        el2n_score = compute_el2n_score(outputs, labels)
        scores.append(el2n_score)
    scores = np.concatenate(scores)

    save_dir = os.path.join("el2n", args.net, settings.TIME_NOW)
    save_path = os.path.join(save_dir, "ckpt.npy")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_path, scores)