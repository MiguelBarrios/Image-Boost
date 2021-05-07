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

def advLoss(pred):
    loss =  torch.mean( 1/1000 * (    (1 - torch.log(pred)) ) ) 
    if loss < 0:
        return 0
    return loss

def totalLossG(output, target, loss_adv):
    loss = torch.mean(((output - target)**2))  + loss_adv   
    return loss

# Trained SRCNN Model path, currently trained on 25 epoches
PATH_GENERATOR = 'Project/SRDiscriminator_model.pt'
PATH_DISCRIMINATOR = 'Project/SRGenerator_model.pt'

root = 'Project/DATASETS/'
high = 'HIGHRES/'
low = 'LOWRES/'
progress = 'progress_data/'
testing = 'testing_data/'
training = 'training_data/'
data = 'data/'

# training data path
data_training_high = root + high + training
data_training_low = root + low + training

# set one data path
data_progress_low = root + low + progress
data_progress_high = root + high + progress

# testing data paths
data_testing_high = root + high + testing
data_testing_low = root + low + testing

# Number of worker threads for dataloader
workers = 2

# Number of GPUs available.
ngpu = 0

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


##### Create Data loaders #####
batch_size = 32

# High res training image dataloader
dataloader_hr_training = loadData(data_training_high, batch_size)

# Low res training image dataloader
dataloader_lr_training = loadData(data_training_low, batch_size)

# Set one low res imgage dataloader
dataloader_low_progress = loadData(data_progress_low, 4)

# Set one high res image dataloader
dataloader_high_progress = loadData(data_progress_high, 4)

# High res testing image dataloader
dataloader_high_testing = loadData(data_testing_high, 32)

# Low res testing image dataloader
dataloader_low_testing = loadData(data_testing_low, 32)

# sample used for traking progresss during training
sample = next(iter(dataloader_low_progress))
##############################################################


total_epochs = 22



"""
# Create the generator
netG = Generator(ngpu).to(device)
netG = netG.to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)

netD = Discriminator(ngpu)

netD = netD.to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)
"""

# Load models
netG, optimizerG = loadSavedModel(PATH_GENERATOR)
netD, optimizerD = loadSavedModel(PATH_DISCRIMINATOR)

# Initialize Discrimiator Loss function
criterion_d = nn.BCELoss()


# labels
real_label = 1.
fake_label = 0.

# Initialize BCELoss function
criterion = nn.MSELoss(reduction='sum')

# list of progress images, after they have been run through G, at each iter
img_list = []

# Generator/Discriminator total loss after each batch
G_losses = []
D_losses = []

# Initialize BCELoss function
criterion = nn.MSELoss()

# Setup Adam optimizers for both G and D
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

batches = len(dataloader_lr_training)

def train(num_epoches, total_epochs):
    # Regular training only generator
    print("Starting Training Loop...")
    for epoch in range(num_epoches):
        i = 0 
        for data_high, data_low in zip(dataloader_hr_training, dataloader_lr_training):
            # Train Discriminator with high Res images
            netD.zero_grad()
            # create batch of high-res images
            real_cpu = data_high[0].to(device)
            batch_size = real_cpu.size(0)
            # create vector of real labels 
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion_d(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
            ######   Generate SR Images #################
            fake = netG(real_cpu) 
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # adversarial loss
            pred_D = output.detach()
            loss_adv = advLoss(pred_D)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion_d(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            error_d = errD_real + errD_fake
            # Update Discriminator
            optimizerD.step()
            ##############################################
            #####     Train Generator    #################
            ##############################################
            netG.zero_grad()
            real_cpu = data_low[0].to(device)
            output = netG(real_cpu) 
            # Calculate loss on all-real batch
            #error_g = criterion_g(output, data_high[0])
            error_g = totalLossG(output, data_high[0], loss_adv)
            # Calculate gradient for G
            error_g.backward()
            D_G_z2 = output.mean().item()
            # update Generator
            optimizerG.step()
            # Save Losses for plotting later
            G_losses.append(error_g.item())
            D_losses.append(error_d.item())
            i = i + 1
            # Check how the generator is doing by saving G's output on fixed_noise
            if(epoch % 25 == 0):
                print("epoch: {}/{} batch:{}/{} error G: {} Error D: {}".format(epoch, num_epoches, i, batches, error_g.item(), error_d.item()))
                torch.save({
                        'epoch': total_epochs,
                        'model_state_dict': netG.state_dict(),
                        'optimizer_state_dict': optimizerG.state_dict(),
                        'loss': error_g,
                        }, PATH_GENERATOR)
                torch.save({
                        'epoch': total_epochs,
                        'model_state_dict': netD.state_dict(),
                        'optimizer_state_dict': optimizerD.state_dict(),
                        'loss': error_d,
                        }, PATH_DISCRIMINATOR)
                with torch.no_grad():
                    cur = sample[0].to(device)
                    fake = netG(cur).detach().cpu()
                    #https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        total_epochs = total_epochs + 1
    print("Finished")

#saveImages(img_list, 'generatedImages/SRRGANN/final/')

"""
gen_images_progress = []
for data_high, data_low in zip(dataloader_high_progress, dataloader_low_progress):
    with torch.no_grad():
        cur = data_low[0].to(device)
        fake = netG(cur).detach().cpu()
        gen_images.append([fake,data_high[0]])
saveImages(gen_images_progress, 'generatedImages/SRRGANN/final/set1/')
"""
"""
gen_images_testing = []
for data_high, data_low in zip(dataloader_high_testing, dataloader_low_testing):
    with torch.no_grad():
        cur = data_low[0].to(device)
        fake = netG(cur).detach().cpu()
        gen_images.append([fake,data_high[0]])
saveImages(gen_images_testing, 'generatedImages/SRRGANN/final/testing/')
"""