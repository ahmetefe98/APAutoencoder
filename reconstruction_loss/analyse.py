import json
import numpy as np
import operator
import statistics
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x = np.concatenate((x_train,x_test))
y = np.concatenate((y_train,y_test))

#ids of the images of each class
a = []
for i in range(100):
    a.append(np.where(y==i)[0])


print('autoencoder:')
with open('./autoencoder/a32.txt', 'r') as f:
    a32 = json.load(f)

print('min: ', min(a32))
print('mean: ', statistics.mean(a32))
print('median: ', statistics.median(a32))
print('max: ', max(a32))

norm_class = []
for i in a:
    norm = []
    for j in i:
        norm.append(a32[j])
    norm_class.append(statistics.mean(norm))

indexed_norm = list(enumerate(norm_class))
top5 = sorted(indexed_norm, key=operator.itemgetter(1))[:5]
worst5 = sorted(indexed_norm, key=operator.itemgetter(1))[-5:]
print('top5 class: ', top5)
print('worst5 class: ', worst5)


print('pca:')
with open('./pca/pca_list_of_norm.txt', 'r') as f:
    pca_list_of_norm = json.load(f)

print('min: ', min(pca_list_of_norm))
print('mean: ', statistics.mean(pca_list_of_norm))
print('median: ', statistics.median(pca_list_of_norm))
print('max: ', max(pca_list_of_norm))

norm_class = []
for i in a:
    norm = []
    for j in i:
        norm.append(pca_list_of_norm[j])
    norm_class.append(statistics.mean(norm))

indexed_norm = list(enumerate(norm_class))
top5 = sorted(indexed_norm, key=operator.itemgetter(1))[:5]
worst5 = sorted(indexed_norm, key=operator.itemgetter(1))[-5:]
print('top5 class: ', top5)
print('worst5 class: ', worst5)