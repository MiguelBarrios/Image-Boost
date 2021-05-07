import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn as nn
from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torch.onnx as onnx
import torchvision.models as models
from PIL import Image
from torchvision.utils import save_image
from helper import *
from ANN import *

def loadData(directory, batch_size):
    dataset = dset.ImageFolder(root=directory,
                               transform=transforms.Compose([ 
                               transforms.Grayscale(num_output_channels=1),
                               transforms.Resize((256,256)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5)),]))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=False, num_workers=workers)

# custom weights initialization called on netG and netD
#https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
#https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def plotGeneratedImages(img_list):
    #%%capture
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())
    
# Plot some training images
#https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def plotBatch(dataloader, dim):
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(dim,dim))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

def plotBatch2(real_batch, dim):
    plt.figure(figsize=(dim,dim))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

def saveImages(img_list, directory):
    DATA_GENERATED = directory
    num_gen = len(img_list)
    for generation in range(len(img_list)):
        gen_dir = DATA_GENERATED + str(generation) + '/'
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)
        for i in range(len(img_list[generation])):
            img = img_list[generation][i]
            output_path = DATA_GENERATED + str(generation) + '/ '+ str(i) + '.png'
            save_image(img, output_path)

def displayProgress(dim,progress_list):
    #%%capture
    fig = plt.figure(figsize=(dim,dim))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in progress_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())

def PSNR(img1,img2):
    mse = torch.sum((img1 - img2) ** 2)
    psnr =  20 * torch.log10(255.0 / torch.sqrt(mse))
    return psnr, mse

def RMSELoss(img1,img2):
    return torch.sqrt(torch.mean((img2-img2)**2))

def PixelLoss(output, target):
    pixelLoss = torch.mean((output - target)**2 / (128*128))
    return pixelLoss

