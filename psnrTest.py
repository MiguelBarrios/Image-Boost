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

# Trained SRCNN Model path, 20 epochs
PATH = 'Project/SRCNN_model.pt'
PATH_GENERATOR = 'Project/SRDiscriminator_model.pt'
PATH_DISCRIMINATOR = 'Project/SRGenerator_model.pt'

root = 'Project/DATASETS/'
high = 'HIGHRES/'
low = 'LOWRES/'
progress = 'progress_data/'
testing = 'testing_data/'
training = 'training_data/'
data = 'data/'

# Number of worker threads for dataloader
workers = 2


# Number of GPUs available.
ngpu = 0

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


##### Create Data loaders #####
batch_size = 32

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

### Calc Metrics for Testing Dataset
def test1(d1,d2):
	psnr_t = 0
	mse_t = 0
	for data_low, data_high in zip(d1, d2):
	    with torch.no_grad():
	        output = netG(data_low[0])
	        psnr, mse = PSNR(output, data_high[0])
	        psnr_t = psnr_t + psnr
	        mse_t = mse_t + mse
	        print("PSNR: {} MSE: {}".format(psnr, mse))
	psnr_t = psnr_t / len(d1)
	mse_t = mse_t /len(d1)
	print("AVG: PSNR: {} AVG: MSE: {}".format(psnr_t, mse_t))

netG, optimizerG = loadSavedModel(PATH_SRCNN_1)
netG_2, x = loadSavedModel(PATH_GENERATOR)
netD, x = loadSavedModel(PATH_DISCRIMINATOR)

#test1(data_progress_low, data_progress_high)
#test1(data_testing_low, data_testing_high)

