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

#Implementation is based on: 
#http://snappishproductions.com/blog/2018/01/03/class-activation-mapping-in-pytorch.html

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(), 
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

display_transform = transforms.Compose([
   transforms.Resize((224,224))])

data_dir = 'rock_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms['val'])
                  for x in ['train', 'val', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()

def image_loader(image):
    """load image, returns cuda tensor"""
    image = image.float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.to(device)  #assumes that you're using GPU

def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]

# Load trained model 
model = torch.load('final_resnet.tar', map_location='cpu')
model.to(device)
model.eval()

# get the weights of the last conv layer
final_layer = model._modules.get('layer4')
activated_features = SaveFeatures(final_layer)

# create directory to save cam images and choose name here: 
directory_dist = 'camtest'
if not exists(directory_dist):
	print('New Distination Directory is Created to Save Filtered Files')
	mkdir(directory_dist)

else: 
	print('Distination Directory Already Exists')

# iterate over images, generate CAMs, and save in folder accordingly

for i, (image_original, classes) in tqdm(enumerate(image_datasets['test']), total=dataset_sizes['test']):

	#save original image
	filename_original = '%d_label%d_.png' % (i, classes)
	path_original = directory_dist + '/' + filename_original
	torchvision.utils.save_image(image_original, path_original, nrow=4)

	#format image and run forward pass
	image = image_loader(image_original)
	prediction = model(image)
	pred_probabilities = F.softmax(prediction).data.squeeze()

	# get the weights of the fc layer
	weight_softmax_params = list(model._modules.get('fc').parameters())
	weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

	# compute the class probability and index
	probability = topk(pred_probabilities,1)[0].item()
	class_idx = topk(pred_probabilities,1)[1].item()

	# compute the cam image 
	overlay = getCAM(activated_features.features, weight_softmax, class_idx)

	#Create filename and path and save image 
	filename = '%d_GT%d_Pred_%0.3f.png' % (i, classes, class_idx)
	path = directory_dist + '/' + filename
	image = image.squeeze() 
	figure, axes = plt.subplots()
	figure = plt.figure()
	axes = figure.add_subplot(111)
	axes.imshow(display_transform(transforms.ToPILImage()(image)))
	axes.imshow(skimage.transform.resize(overlay[0], image.shape[1:3]), alpha=0.5, cmap='jet');
	figure.savefig(path)



