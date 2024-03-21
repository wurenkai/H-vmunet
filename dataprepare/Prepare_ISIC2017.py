# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:15:43 2019
@author: Reza Azad
"""

from __future__ import division
import numpy as np
import scipy.io as sio
import scipy.misc as sc
import glob

# Parameters
height = 256
width  = 256
channels = 3

############################################################# Prepare ISIC 2017 data set #################################################
Dataset_add = '../data/dataset_isic17/'
Tr_add = 'ISIC-2017_Training_Data'

Tr_list = glob.glob(Dataset_add+ Tr_add+'/*.jpg')
# It contains 2594 training samples
Data_train_2017    = np.zeros([2000, height, width, channels])
Label_train_2017   = np.zeros([2000, height, width])

print('Reading ISIC 2017')
for idx in range(len(Tr_list)):
    print(idx+1)
    img = sc.imread(Tr_list[idx])

   
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
    Data_train_2017[idx, :,:,:] = img

    
    b = Tr_list[idx]    
    a = b[0:len(Dataset_add)]
    b = b[len(b)-16: len(b)-4] 
    add = (a+ 'ISIC-2017_Training_Part1_GroundTruth/' + b +'.png')    
    img2 = sc.imread(add)
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_train_2017[idx, :,:] = img2    
         
print('Reading ISIC 2017 finished')

################################################################ Make the train and test sets ########################################    
# We consider 1250 samples for training, 150 samples for validation and 600 samples for testing

Train_img      = Data_train_2017[0:1250,:,:,:]
Validation_img = Data_train_2017[1250:1250+150,:,:,:]
Test_img       = Data_train_2017[1250+150:2000,:,:,:]

Train_mask      = Label_train_2017[0:1250,:,:]
Validation_mask = Label_train_2017[1250:1250+150,:,:]
Test_mask       = Label_train_2017[1250+150:2000,:,:]


np.save('data_train', Train_img)
np.save('data_test' , Test_img)
np.save('data_val'  , Validation_img)

np.save('mask_train', Train_mask)
np.save('mask_test' , Test_mask)
np.save('mask_val'  , Validation_mask)


