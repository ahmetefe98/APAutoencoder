### Advanced practical: Comparison of autoencoder and PCA for dimension reduction on the CIFAR-100 data set ###

+ in this folder are the code for the advanced practial

### Prerequisites ###

To use this code you need:

+ TensorFlow 2.0 or higher
+ Python 3.5-3.7

### Folder strucutre and content  ###

+ "cluster": in this folder is an approch to cluster images with an autoencoder, for clarity the code is split in two files. In functions.py are the functions and imports which are needed, in main.py is the main code. Through the save weights the models must not be fit and only load and compile. Moreover, the figure 6 "Vergleich echte Klasse und vermutetet Klasse CIFAR-100" from page 8 of the report is in a hig resolution in this folder.  
+ "data": in this folder are information about the dataset. The file classnames.py print the 100 classnames with their numbers, this list is also in the file classnames with classnumbers.txt. The file unpickle.py works with the data and create the images from the folder data/images/. To use unpickle.py you need to download and decompress 'CIFAR-100 python version' from https://www.cs.toronto.edu/~kriz/cifar.html . In images/classes there are for all 100 class five example images. In images/superclasses there are for all 20 superclass one examaple image of the classes which are grouped in this superclass.



### Author ###

+ Ahmet Efe, email: im224@stud.uni-heidelberg.de

31.01.2021 Heidelberg, Germany
