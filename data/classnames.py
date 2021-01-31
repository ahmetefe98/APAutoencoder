import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1') #bytes
    return dict

#meta data dict: names of the 100 classes (e.g. b'apple')
meta_data_dict = unpickle('cifar-100-python/meta')
#label_names np.array with the 100 classnames
label_names = np.array(meta_data_dict['fine_label_names'])

for i in range(100):
    print(i, label_names[i])