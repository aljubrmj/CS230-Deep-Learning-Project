from __future__ import print_function, division

from os import listdir, mkdir
from os.path import  isfile, join, splitext, exists 
import sys
import torch
from torch import topk
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.pyplot import imshow
import time
import os
import copy
from tqdm import tqdm
from PIL import Image
import skimage.transform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'rock_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=False, num_workers=4)
              for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['test'].classes

print(class_names)

# Load trained model 
model = torch.load('final_densenet.tar', map_location='cpu')
model.to(device)
model.eval()

running_corrects = 0
confusion_matrix = torch.zeros(len(class_names), len(class_names))
for i, (inputs, classes) in tqdm(enumerate(dataloaders['test']), total=dataset_sizes['test']):
    inputs = inputs.to(device)
    classes = classes.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    running_corrects += (preds.item() == classes.data.item())
    for t, p in zip(classes.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

test_acc = running_corrects/len(image_datasets['test'])
print('Confusion Matrix: ', confusion_matrix)
print('Precision: ', confusion_matrix.diag()/confusion_matrix.sum(1))
print('Recall: ', confusion_matrix.diag()/confusion_matrix.sum(0))
print('Accuracy: ', test_acc)

