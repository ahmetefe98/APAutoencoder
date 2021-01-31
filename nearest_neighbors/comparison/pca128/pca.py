from keras.datasets import cifar100
from sklearn.decomposition import PCA
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
import statistics
from collections import Counter
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
x = pca.transform(x)

a = []
for i in range(100):
    a.append(np.where(y==i)[0][0:10])

list_of_accurarcy = []
list_of_counter_keys = []
list_of_counter_values = []
for i in range(len(a)):
    temp = []
    for n in a[i]:
        query = x[n]
        label = y[n]
        n_neigh = 6
        x = x.reshape(-1, 128)
        query = query.reshape(1,128)
        nbrs = NearestNeighbors(n_neighbors=n_neigh, n_jobs = -1).fit(x)
        distances, indices = nbrs.kneighbors(np.array(query))
        n_label_names = [label_names[y[i]] for i in indices]
        print(n_label_names[0])
        count = 0
        for i in range(1,6):
            if label == y[indices[0][i]]:
                    count += 1
        accuracy = count / 5.
        temp.append(accuracy)
    list_of_accurarcy.append(statistics.mean(temp))
    list_of_counter_keys.append(list(Counter(temp).keys()))
    list_of_counter_values.append(list(Counter(temp).values()))

with open('acc.txt', 'w') as f:
    f.write(json.dumps(list_of_accurarcy))

with open('c_keys.txt', 'w') as f:
    f.write(json.dumps(list_of_counter_keys))

with open('c_values.txt', 'w') as f:
    f.write(json.dumps(list_of_counter_values))
