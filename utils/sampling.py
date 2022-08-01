#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from stringprep import in_table_c9
import numpy as np
from torchvision import datasets, transforms
import csv
import os
from os import path
import urllib.request 
import zipfile

def cifar_noniid(dataset, num_users): 
    num_dataset = len(dataset) 
    idx = np.arange(num_dataset) 
    dict_users={}
    for i in range(num_users):
        dict_users[str(i)] = []

    min_num = 200 
    max_num = 800 
    count =0
    # divide and assign 
    for i in range(num_users): 
        if len(idx) > 0 and i < 99: 
            rand_set = set(np.random.choice(idx, np.random.randint(min(min_num,len(idx))-1, min(len(idx),max_num)), replace=False)) 
            idx = list(set(idx) - rand_set) 
            #dict_users[i] = rand_set
            dict_users[str(i)] = rand_set 
        if i==99: 
            rand_set = set(np.random.choice(idx, len(idx), replace=False)) 
            idx = list(set(idx)-rand_set) 
            #dict_users[i] = rand_set 
            dict_users[str(i)] = rand_set 
        count = count + len(dict_users[str(i)])
  
    print(f"Total number of dataset images owned by clients : {count}") 
             
    return dict_users

def file_parser(line, is_train=True):
  if is_train:
    user_id, image_id, class_id = line
    return user_id, image_id, class_id
  else:
    image_id, class_id = line
    return image_id, class_id

def dirichlet_distribution(args):    
  url="http://storage.googleapis.com/gresearch/federated-vision-datasets/cifar10_v1.1.zip"
  dir=os.getcwd()+"/cifar10_csv"

  try:
    os.mkdir(dir)
  except:
    print("Folder already exist")
    
  urllib.request.urlretrieve(url, dir+"/cifar.zip")
  with zipfile.ZipFile(dir+"/cifar.zip","r") as zip_ref:
      zip_ref.extractall(dir)
  
  alpha=str("{:.2f}".format(args.alpha))
  train_file=dir+"/federated_train_alpha_"+alpha+".csv"

  print('Train file: %s' % train_file)
  if not path.exists(train_file):
    print('Error: file does not exist.')
    return

  dict_users={}
  for i in range(args.num_users):
      dict_users[str(i)] = []
      
  with open(train_file) as f:
    reader = csv.reader(f)
    next(reader)  # skip header.
    for line in reader:
      user_id, image_id, class_id = file_parser(line, is_train=True)
      dict_users[user_id].append(int(image_id))
  return dict_users

