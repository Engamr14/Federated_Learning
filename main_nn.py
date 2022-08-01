#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from tabulate import tabulate

from utils.options import args_parser
from models.Nets import LeNet5, mobilenet_v3, cnn_cifar10

import logging

IMAGE_SIZE = 32

def test(net_g, data_loader, epoch):
    # testing
    net_g.eval()
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return accuracy, test_loss


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    logging.getLogger('matplotlib.font_manager').disabled = True

    torch.manual_seed(args.seed)

    # load dataset 
    if args.dataset == 'cifar':
        train_transform = transforms.Compose([    
            transforms.RandomCrop(IMAGE_SIZE, padding=4),    
            transforms.RandomHorizontalFlip(),    
            transforms.ToTensor(),    
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),    ])  
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, transform=train_transform, target_transform=None, download=True)

        test_transform = transforms.Compose([    
            transforms.ToTensor(),    
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),    ])       
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, transform=test_transform, target_transform=None, download=True)
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.model == 'lenet5':
        net_glob = LeNet5(args=args).to(args.device)
    elif args.model == 'mobilenet':
        net_glob = mobilenet_v3(args=args).to(args.device)
    elif args.model == 'cnn_cifar':
        net_glob = cnn_cifar10(num_classes=args.num_classes, num_channels=args.num_channels, model_args=args)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    if args.gpu != -1:
        net_glob.cuda()

    # training
    optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)

    test_acc_list, test_idxs, test_loss_list, headers  = ["Test Accuracy"], [], ["Test Loss"], [""]
    list_loss = []
    net_glob.train()
    for epoch in range(args.epochs):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net_glob(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain avg loss:{}\n\n'.format(loss_avg))
        list_loss.append(loss_avg)

        if epoch == 10 or epoch % 50 == 0:
            #if epoch != 0:
                print('test on', len(dataset_test), 'samples')
                test_acc, test_loss = test(net_glob, test_loader, epoch)
                test_acc_list.append(test_acc)
                test_idxs.append(epoch+1)
                test_loss_list.append(test_loss)

                net_glob.train()


    # plot loss curve
    plt.figure()
    plt.plot(range(len(list_loss)), list_loss)
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.title('Training loss ')
    plt.savefig('./log/nn/nn_loss_curve_{}_{}_ep_{}_mom_{}_lr_{}.png'.format(args.dataset, args.model, args.epochs, args.momentum, args.lr))

    test_acc, test_loss = test(net_glob, test_loader, args.epochs)
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)
    test_idxs.append(args.epochs)

    for i in test_idxs:
        headers.append("R{}".format(i))

    data = []
    data.append(test_acc_list)
    data.append(test_loss_list)
    print(tabulate(data, headers=headers))

    logging.basicConfig(filename="./log/nn/nn.log", 
					format='%(message)s', 
					level=logging.DEBUG)
    logger=logging.getLogger() 
    logger.info('\nnn_{}_{}\nMomentum: {}\tLearning rate: {}\n'.format(args.dataset, args.model, args.momentum, args.lr))
    logger.info('--------------------------------------')
    logger.info(tabulate(data, headers=headers))
    logger.info('--------------------------------------')

    #plot accuracy
    plt.figure()
    plt.plot(test_idxs, test_acc_list, '-o')
    plt.xlabel('epochs')
    plt.ylabel('test accuracy')
    plt.title('Test accuracy ')
    plt.savefig('./log/nn/nn_acc_{}_{}_ep_{}_mom_{}_lr_{}.png'.format(args.dataset, args.model, args.epochs, args.momentum, args.lr))