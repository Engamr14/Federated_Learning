#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
from torch.utils.data import dataset
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from tabulate import tabulate
import torch.optim as optim

from utils.sampling import cifar_noniid, dirichlet_distribution
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import LeNet5, mobilenet_v3, cnn_cifar10
from models.Fed import FedAvgWeighted
from models.test import test_img
import logging

IMAGE_SIZE = 32

if __name__ == '__main__':
    # parse args
    args = args_parser()
    np.random.seed(args.seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    logging.getLogger('matplotlib.font_manager').disabled = True

    # load dataset and split users

    if args.dataset == 'cifar':
        train_transform = transforms.Compose([    
            transforms.RandomCrop(IMAGE_SIZE, padding=4),    
            transforms.RandomHorizontalFlip(),    
            transforms.ToTensor(),    
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),    ])        
        test_transform = transforms.Compose([    
            transforms.ToTensor(),    
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),    ])       

        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=train_transform)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=test_transform)

        if args.distribution == 'non-iid':
            dict_users = cifar_noniid(dataset_train, args.num_users)
        elif args.distribution == 'dirichlet':
            dict_users = dirichlet_distribution(args)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    print('Client split IID: {}'.format(args.distribution))

    # build model
    if args.model == 'lenet5':
        net_glob = LeNet5(args=args).to(args.device)
    elif args.model == 'mobilenet':
        net_glob = mobilenet_v3(args=args).to(args.device)
    elif args.model == 'cnn_cifar':
        net_glob = cnn_cifar10(num_classes=args.num_classes, num_channels=args.num_channels, model_args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    
    if args.gpu != -1:
        net_glob.cuda()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    test_acc_list, train_acc_list, test_idxs, headers = ["Test set acc."], ["Train set acc."], [], [""]


    w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        print("\nSelected users indexes {}\n".format(idxs_users))
        
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[str(idx)])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals[idx] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))
            
        # update global weights
        w_glob = FedAvgWeighted(w_locals, dict_users, len(dataset_train))

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        #intermediate training
        if iter == 10 or iter % 50 == 0:
            #if epoch != 0:
                print('test on {} test samples and {} train samples'.format(len(dataset_test), len(dataset_train)))

                train_acc, train_loss = test_img(net_glob, dataset_train, args)
                test_acc, test_loss = test_img(net_glob, dataset_test, args)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                print("Training accuracy: {:.2f}".format(train_acc))
                print("Testing accuracy: {:.2f}".format(test_acc))
                test_idxs.append(iter+1)


    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./log/fed/fed_loss_curve_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.distribution))

    # testing
    net_glob.eval()
    train_acc, train_loss = test_img(net_glob, dataset_train, args)
    test_acc, test_loss = test_img(net_glob, dataset_test, args)
    test_idxs.append(args.epochs)
    test_acc_list.append(test_acc)
    train_acc_list.append(train_acc)
    for i in test_idxs:
        headers.append("R{}".format(i))

    data = []
    data.append(test_acc_list)
    data.append(train_acc_list)
    print(tabulate(data, headers=headers))
    print("Training accuracy: {:.2f}".format(train_acc))
    print("Testing accuracy: {:.2f}".format(test_acc))


    logging.basicConfig(filename="./log/fed/fed.log", 
                    format='%(message)s', 
                    level=logging.DEBUG)
    logger=logging.getLogger() 
    logger.info('\nFed_{}_{}_{}\nLocal epochs: {}\nSelected clients: {}\nMomentum: {}\tLearning rate: {}'.format(args.dataset, args.model, args.fed, args.local_ep, args.frac*args.num_users, args.momentum, args.lr))
    if(args.fed == 'FedAVGM'):
        logger.info('Server momentum: {}'.format(args.momentum))
    if(args.distribution == 'dirichlet'):
        logger.info('Distribution: {}\talpha:{}'.format(args.distribution, args.alpha))
    else:
        logger.info('Distribution: {}'.format(args.distribution))

    logger.info('--------------------------------------')
    logger.info(tabulate(data, headers=headers))
    logger.info('--------------------------------------')

