# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""
import json
import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

import calculators

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# assert device == "cuda"

random_seed = 43
torch.manual_seed(random_seed)
total_ind = 0
calculator = None


def train(epoch, data_ledger=None):
    global total_ind
    # global calculator
    start = time.time()
    net.train()
    curr_ind = 0
    print('Start Training using args:')
    print(args)
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()

        # Hard-coded keep=0.5
        sample_ind = sample_example_ind(net, images, labels, keep=0.5, sampling=args.sbp)

        if not sample_ind:
            continue
        curr_ind += len(sample_ind)
        total_ind += len(sample_ind)
        if data_ledger is not None:
            for ind in sample_ind:
                data_ind = int(batch_index * args.b + ind)
                if data_ind in data_ledger:
                    data_ledger[batch_index * args.b + ind].append(epoch)
                else:
                    data_ledger[batch_index * args.b + ind] = [epoch]

        images_selected = images[sample_ind]
        labels_selected = labels[sample_ind]

        outputs_selected = net(images_selected)
        loss = loss_function(outputs_selected, labels_selected)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)


        print('Training Epoch: {epoch} [{trained_samples} (used {used_samples})/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset),
            used_samples=curr_ind
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        #update number of used examples
        writer.add_scalar('Used/Total', total_ind, n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


# def sample_example_ind(calculator, net, images, labels, sampling=True):
#     if not sampling:
#         return np.arange(0,images.shape[0]).tolist()
#     # loss_total_check = loss_function(net(images), labels)
#     exp_selected = []
#     # cum_loss = 0
#     for exp_ind in range(images.shape[0]):
#         with torch.no_grad():
#             # net(images)
#             img, label = images[exp_ind].unsqueeze(0), labels[exp_ind].unsqueeze(0)
#             output = net(img)
#             example_loss = loss_function(output, label)
#             # this is a test, not actually doing sampling base on
#             # if exp_ind%3==0:
#             #     exp_selected.append(exp_ind)
#
#             loss_val = example_loss.cpu().data.item()
#             calculator.append(loss_val)
#             prob = calculator.calculate_probability(loss_val)
#             if np.random.rand() < prob:
#                 exp_selected.append(exp_ind)
#         # cum_loss += example_loss.cpu().data
#
#     # assert (loss_total_check.cpu() - cum_loss <1e-5)
#
#     return exp_selected

# Code Adapted From
# https://github.com/mosaicml/composer/blob/dev/composer/algorithms/selective_backprop/selective_backprop.py
def sample_example_ind(model, images, labels, keep, sampling=True):
    if not sampling:
        return np.arange(0,images.shape[0]).tolist()

    with torch.no_grad():
        N = input.shape[0]

        # Get per-examples losses
        out = model(images)
        losses = loss_function(out, labels, reduction="none")

        # Sort losses
        sorted_idx = torch.argsort(losses)
        n_select = int(keep * N)

        # Sample by loss
        percs = np.arange(0.5, N, 1) / N
        probs = percs ** ((1.0 / keep) - 1.0)
        probs = probs / np.sum(probs)
        select_percs_idx = np.random.choice(N, n_select, replace=False, p=probs)
        select_idx = sorted_idx[select_percs_idx]

    return select_idx


@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-sbp', action='store_true', default=False, help='run with selective backprop')
    parser.add_argument('-beta', type=int, default=1, help='selective backprop param')
    args = parser.parse_args()

    net = get_network(args)

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=False
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    num_samples = len(cifar100_training_loader.dataset)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    turn_on_sbp = False
    if args.sbp:
        turn_on_sbp = True
        args.sbp = False

    data_ledger = {}

    calculator = calculators.RelativeProbabilityCalculator(device, beta=args.beta)

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        if epoch > ((settings.EPOCH)//2) and turn_on_sbp:
            print("Half way through training, turnning on SBP")
            args.sbp = True

        train(epoch, data_ledger)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    print("="*20+"Training Data stats:"+"="*20)
    print(data_ledger)
    jsonfilename = "./data_ledger_"+args.net
    if args.sbp:
        jsonfilename += "_sbp"

    with open(jsonfilename+".json", 'w+') as json_file:
        json.dump(data_ledger, json_file)

    writer.close()
