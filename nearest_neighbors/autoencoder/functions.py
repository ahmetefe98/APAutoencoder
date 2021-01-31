import matplotlib.pyplot as plt
import numpy as np
import pickle
from random import randrange

import keras
import keras.backend as K
from keras.datasets import cifar100
from keras.layers import Input, BatchNormalization, Conv2D, MaxPool2D, UpSampling2D, Activation
from keras.models import Model
from keras.objectives import mean_squared_error
from keras.optimizers import SGD
from sklearn.neighbors import NearestNeighbors

## Convolution block of 2 layers
def create_block(input, chs):
    x = input
    for i in range(2):
        x = Conv2D(chs, 3, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
    return x

def general_ae():
    input = Input((32,32,3))
    # Encoder
    block1 = create_block(input, 32)
    x = MaxPool2D(2)(block1)
    block2 = create_block(x, 64)
    x = MaxPool2D(2)(block2)
    #Middle
    middle = create_block(x, 128)
    # Decoder
    up1 = UpSampling2D((2,2))(middle)
    block3 = create_block(up1, 64)
    up2 = UpSampling2D((2,2))(block3)
    block4 = create_block(up2, 32)
    # output
    x = Conv2D(3, 1)(up2)
    output = Activation("sigmoid")(x)
    return Model(input, middle), Model(input, output)

## loss function for using in autoencoder models
def loss_function(y_true, y_pred): 
    mses = mean_squared_error(y_true, y_pred)
    return K.sum(mses, axis=(1,2))

#function from https://www.cs.toronto.edu/~kriz/cifar.html to unpickle the files 
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1') #bytes
    return dict

def get_label_names():
    #meta data dict: names of the 100 classes (e.g. b'apple')
    meta_data_dict = unpickle('cifar-100-python/meta')
    #label_names np.array with the 100 classnames
    label_names = np.array(meta_data_dict['fine_label_names'])
    return label_names