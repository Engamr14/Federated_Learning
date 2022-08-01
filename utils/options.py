#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

#TODO removeunused params 

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=150, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--fed', type=str, default='FedAVGWeighted', help='alogorithm used to update global model: FedAVGWeighted')
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0, help="SGD momentum (default: 0.9 for centralized model, 0 for federated model)")
    parser.add_argument('--weight_decay', type=float, default=4e-4)

    # model arguments
    parser.add_argument('--model', type=str, default='cnn_cifar', help='model name: lenet5 (default), cnn_cifar, mobilenet')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, group_norm, or None")
    parser.add_argument('--variant', type=str, default='small', help="MobileNetV3 variant: small or large")
    parser.add_argument('--pretrained', type=str, default='store_false', help="Pretrain")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--distribution', type=str, default="dirichlet", help='iid, non-iid, dirichlet')
    parser.add_argument('--alpha', type=float, default=0.5, help="alpha values for Dirichlet distribution [0, 0.05, 0.1, 0.20, 0.5, 1, 10, 100]")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()
    return args
