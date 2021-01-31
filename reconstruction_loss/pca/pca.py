from keras.datasets import cifar100
from sklearn.decomposition import PCA
import numpy as np
import pickle
import json

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

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
label_names = get_label_names()

x = np.concatenate((x_train,x_test))
y = np.concatenate((y_train,y_test))
x = x/255.0
x = x.reshape(-1,32,32,3)
x = x.reshape(-1,3072)

pca = PCA(128)
pca.fit(x)
middle_image = pca.transform(x)
output_image = pca.inverse_transform(middle_image)

list_of_norm = []
for i in range(len(x)):
    diff = x[i] - output_image[i]
    norm = sum(abs(diff))
    list_of_norm.append(norm)

with open('pca_list_of_norm.txt', 'w') as f:
    f.write(json.dumps(list_of_norm))
