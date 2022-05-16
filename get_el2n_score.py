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
    args = parser.parse_args()

    #load model
    # recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
    # if not recent_folder: raise Exception('no recent folder were found')
    # checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
    # net = get_network(args)
    # net.load_state_dict(torch.load(checkpoint_path))
    net = models.resnet18(pretrained=True)

    # load training data
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    # compute el2n score
    scores = []
    num_batches = len(cifar100_training_loader.dataset) // args.b
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        print(f'computing el2n score batch {batch_index} of {num_batches}')
        outputs = net(images)
        el2n_score = compute_el2n_score(outputs, labels)
        scores.append(el2n_score)
    scores = np.concatenate(scores)

    save_dir = os.path.join("el2n", args.net, settings.TIME_NOW)
    save_path = os.path.join(save_dir, "ckpt.npy")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_path, scores)