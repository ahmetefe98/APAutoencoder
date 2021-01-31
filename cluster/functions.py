import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import keras
import keras.backend as K
from keras.layers import BatchNormalization, Conv2D , MaxPool2D, UpSampling2D, Flatten
from keras.layers import  Activation,  Input, Dense, Dropout
from keras.datasets import cifar100
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adadelta, SGD
from keras.objectives import mean_squared_error

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

## Convolution block of 2 layers
def create_block(input, chs):
    x = input
    for i in range(2):
        x = Conv2D(chs, 3, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
    return x

## function used for visualizing original and reconstructed images of the autoencoder model
def showOrigDec(orig, dec, num=10): 
    n = num
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(orig[300*i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i +1 + n)
        plt.imshow(dec[300*i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

## function used for visualizing the predicted and true labels of test data
def show_test(m, d, x_test_final, y_test_final, dict):
    plt.figure(figsize =(40,8))
    for i in range(5):
        ax = plt.subplot(1, 5, i+1)
        test_image = np.expand_dims(d[1810*i+5], axis=0)
        test_result = m.predict(test_image)
        plt.imshow(x_test_final[1810*i+5])
        index = np.argsort(test_result[0,:])
        plt.title("Pred:{}, True:{}".format(dict[index[9]], dict[y_test_final[1810*i+5][0]]))
    plt.show()

## function used for creating a classification report and confusion matrix
def report(predictions, y_test_one_hot, dict): 
    cm=confusion_matrix(y_test_one_hot.argmax(axis=1), predictions.argmax(axis=1))
    print("Classification Report:\n")
    cr=classification_report(y_test_one_hot.argmax(axis=1),
                                predictions.argmax(axis=1), 
                                target_names=list(dict.values()))
    print(cr)
    plt.figure(figsize=(12,12))
    sns.heatmap(cm, annot=True, xticklabels = list(dict.values()), yticklabels = list(dict.values()), fmt="d")
    plt.show()

## loss function for using in autoencoder models
def loss_function(y_true, y_pred): 
    mses = mean_squared_error(y_true, y_pred)
    return K.sum(mses, axis=(1,2))

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

def classifier_dense(inp):
    input = Input((inp.shape[1], inp.shape[2], inp.shape[3]))
    x = Flatten()(input)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.64)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(100, activation='softmax')(x)
    return Model(input, output)
