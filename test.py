import scipy.io as sio
import torch
import numpy as np
import matplotlib.pyplot as plt
from unet import Unet
import torch.nn as nn
import nibabel as nib
#import pydicom
import hdf5storage as hdf
import os
#import math
from unet import Unet
import skimage.io as io


ori_path = './te_da'
filenames=os.listdir(ori_path)



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = Unet()
net.load_state_dict(torch.load('./model/model_epoch_24.pth'))
net.to(device)


for i in range(len(filenames)):
    
    data_in = hdf.loadmat(os.path.join('./te_da',filenames[i]))['dwi']
    data_in = np.transpose(data_in,(2,0,1))
    xs = np.zeros((data_in.shape[0],1,240,240))
    xs[:,0,:,20:220] = data_in

    inputs = torch.from_numpy(xs).float()
    inputs = inputs.to(device)
    net.eval()                   
    with torch.no_grad():
        outputs = net(inputs)
        temp1 = np.squeeze(outputs.cpu().detach().numpy())
        temp2 = np.argsort(-temp1,axis=1)
        temp3 = np.squeeze(temp2[:,0,:,:])
        res_data = np.transpose(temp3[:,:,20:220],(1,2,0))       

    sio.savemat(os.path.join('./te_result',filenames[i]), {'dwi':res_data})

    
      

