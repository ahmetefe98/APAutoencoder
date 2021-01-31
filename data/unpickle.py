import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

#function from https://www.cs.toronto.edu/~kriz/cifar.html to unpickle the files 
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1') #bytes
    return dict

if os.path.exists('./cifar-100-python/'):
    pass
else:
    sys.exit("Please download 'CIFAR-100 python version' from https://www.cs.toronto.edu/~kriz/cifar.html")

############### meta ###############
#meta data dict: names of the 100 classes (e.g. b'apple')
meta_data_dict = unpickle('cifar-100-python/meta')
#coarse_names np.array with the 20 superclassnames
coarse_names = np.array(meta_data_dict['coarse_label_names'])
#label_names np.array with the 100 classnames
label_names = np.array(meta_data_dict['fine_label_names'])

############### train ###############
#train data set: 50000 images
train_data_dict = unpickle('cifar-100-python/train')
#filenames of all 50000 images (e.g. 'bos_taurus_s_000507.png')
train_filenames = train_data_dict['filenames']
#convert to a np.array
train_filenames = np.array(train_filenames)
#batch label: 'training batch 1 of 1'
train_batch_label = train_data_dict['batch_label']
#fine labels: label number of all 50000 images (e.g. 19 (for the name label_names[19]))
train_fine_labels = train_data_dict['fine_labels']
#convert to a np.array
train_fine_labels = np.array(train_fine_labels)
#coarse labels: coarse number of all 50000 images (e.g. 19 (for the name coarse_names[19]))
train_coarse_labels = train_data_dict['coarse_labels']
#convert to a np.array
train_coarse_labels = np.array(train_coarse_labels)
#data: 50000 arrays with red, green and blue values for each 32x32 pixel of the 
#image, so every array has 3072 numbers
train_data = train_data_dict['data']
#convert: the 3072 numbers in each array split up in  32 arrays with 32 numbers 
train_data = train_data.reshape((len(train_data), 3, 32, 32))
#convert: 50000 arrays with 32 arrays, which has 32 triple with the red, green 
#and blue value of the pixel
train_data = np.rollaxis(train_data, 1, 4)

############### test ###############
#test data set: 10000 images
test_data_dict = unpickle('cifar-100-python/test')
#filenames of all 10000 images (e.g. 'volcano_s_000012.png')
test_filenames = test_data_dict['filenames']
#convert to a np.array
test_filenames = np.array(test_filenames)
#batch label: 'training batch 1 of 1'
test_batch_label = test_data_dict['batch_label']
#fine labels: label number of all 10000 images (e.g. 49 (for the name label_names[49]))
test_fine_labels = test_data_dict['fine_labels']
#convert to a np.array
test_fine_labels = np.array(test_fine_labels)
#coarse labels: coarse number of all 50000 images (e.g. 19 (for the name coarse_names[19]))
test_coarse_labels = train_data_dict['coarse_labels']
#convert to a np.array
test_coarse_labels = np.array(test_coarse_labels)
#data: 10000 arrays with red, green and blue values for each 32x32 pixel of the 
#image, so every array has 3072 numbers
test_data = test_data_dict['data']
#convert: the 3072 numbers in each array split up in  32 arrays with 32 numbers 
test_data = test_data.reshape((len(test_data), 3, 32, 32))
#convert: 10000 arrays with 32 arrays, which has 32 triple with the red, green 
#and blue value of the pixel
test_data = np.rollaxis(test_data, 1, 4)

#dict with 20 elements, key is the name of the superclass, the elements are a
#list with the names of the classes
super_and_classnames = {'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                    'food containers':	['bottle', 'bowl', 'can', 'cup', 'plate'],
                    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                    'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
                    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
                    'reptiles':	['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']}

#visualize a superclass and plot one image for each class of the superclass
def visualizesuperClass(superclassname, classname):
    f, ax = plt.subplots(1,5)
    f.set_size_inches(20,10)
    f.suptitle(superclassname, fontsize=16)
    for m in range(5):
        label_number = np.where(label_names == classname[m])[0]
        idx = np.where(train_fine_labels == label_number)[0][0]
        ax[m].imshow(train_data[idx])
        ax[m].get_xaxis().set_visible(False)
        ax[m].get_yaxis().set_visible(False)
        ax[m].set_title(classname[m] + " \n " + train_filenames[idx])
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0.5)
    plt.savefig('./images/superclasses/' + superclassname)

#visualize a class and plot five image from the class
def visualizeClass(classname):
    f, ax = plt.subplots(1,5)
    f.set_size_inches(20,10)
    f.suptitle(classname, fontsize=16)
    label_number = np.where(label_names == classname)[0]
    idx = np.where(train_fine_labels == label_number)[0]
    for m in range(5):
        ax[m].imshow(train_data[idx[m]])
        ax[m].get_xaxis().set_visible(False)
        ax[m].get_yaxis().set_visible(False)
        ax[m].set_title(train_filenames[idx[m]])
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0.5)
    plt.savefig('./images/classes/' + classname)

for i in super_and_classnames:
    visualizesuperClass(i, super_and_classnames[i])

for i in super_and_classnames:
    for j in super_and_classnames[i]:
        visualizeClass(j)