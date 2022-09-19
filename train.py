import torch
import numpy as np
import matplotlib.pyplot as plt
import visdom
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import hdf5storage as hdf
import os
from unet import Unet
from torch.nn import BCELoss
import math


batch_size=10
data_in = hdf.loadmat('./tr_da')
x_data = data_in['tr_data']
y_data = data_in['tr_label']
sample_num = x_data.shape[0]

train_iteration     = math.ceil(sample_num//batch_size)


# ------------------模型，优化方法------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = Unet()
print(net)
net.to(device)
net.train()
#optimizer = optim.SGD(net.parameters(), lr=0.001)

#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr = 0.0001)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
loss_fc = torch.nn.CrossEntropyLoss()

# -----------------训练---------------------------------------
#viz = visdom.Visdom()
#loss_win = viz.line(np.arange(2))

def data_get_tr(duan_num, indexx, batch_size, x_da,y_da):
    star = duan_num*batch_size
    endd = (duan_num+1)*batch_size
    indd = indexx[star:endd]
    inputss_x = np.zeros((batch_size,1,240,240))
    inputss_y = np.zeros((batch_size,240,240))    
    inputss_x[:,0,:,20:220] = x_da[indd,:,:]
    inputss_y[:,:,20:220] = y_da[indd,:,:]
                                
    return inputss_x, inputss_y


loss_all = []
iter_count = 0
for epoch in range(25):

    running_loss = 0.0
    tr_loss = 0.0
    idx = np.random.permutation(sample_num)
 
#    scheduler.step()
    for i in range(train_iteration):
        xs, ys = data_get_tr(i,idx,batch_size,x_data,y_data)

        inputs = torch.from_numpy(xs).float() 
        labels = torch.from_numpy(ys).long()
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs.shape
        labels.shape

#        net.train()
        optimizer.zero_grad()
        outputs = net(inputs)
        outp = outputs.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        labe = labels.view(batch_size*240*240)
 #       print(outp.shape)
 #       print(labe.shape)
        loss = loss_fc(outp, labe)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
#        print(str(i)+':'+str(loss.item()))
#        if i>100:
#            torch.save(net.state_dict(), './model2/model_epoch_{:02d}.pth'.format(i))

    tr_loss = running_loss / train_iteration
    print('epoch_{:02d}'.format(epoch)+'loss_{:f}'.format(tr_loss))
    loss_all.append(tr_loss)
 #   viz.line(Y=np.array([tr_loss]), X=np.array([epoch]), update='append', win=loss_win)
    if epoch>15:
        torch.save(net.state_dict(), './model/model_epoch_{:02d}.pth'.format(epoch))

print('Train finish!')
