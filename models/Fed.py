#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvgWeighted(w, dict_users, dataset_len):
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        w_avg[k] = w[0][k] * len(dict_users[str(0)])
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * len(dict_users[str(i)])
        w_avg[k] = torch.div(w_avg[k], dataset_len)
    return w_avg
