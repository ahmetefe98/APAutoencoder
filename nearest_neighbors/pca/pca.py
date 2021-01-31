import numpy as np
import pickle
from random import randrange
import matplotlib.pyplot as plt

from keras.datasets import cifar100
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

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
x = pca.transform(x)

n = randrange(60000)
print(n)
query = x[n]
label = y[n]
n_neigh = 6
x = x.reshape(-1, 128)
query = query.reshape(1,128)
nbrs = NearestNeighbors(n_neighbors=n_neigh, n_jobs = -1).fit(x)
distances, indices = nbrs.kneighbors(np.array(query))
n_label_names = [label_names[y[i]] for i in indices]
closest_images = x[indices]
closest_images = closest_images.reshape(-1,32,32,3)

plt.imshow(query.reshape(32,32,3))
plt.title(label_names[label])
plt.show()
plt.figure(figsize=(20, 6))
for i in range(1, n_neigh):
    # display original
    ax = plt.subplot(1, n_neigh, i+1)
    ax.set_title(n_label_names[0][i])
    plt.imshow(closest_images[i].reshape(32,32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
