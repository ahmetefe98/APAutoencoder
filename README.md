### Advanced practical: Comparison of autoencoder and PCA for dimension reduction on the CIFAR-100 data set ###

+ in this folder are the code for the advanced practial 
+ for more information about the practial please read the report.pdf (german)
+ all used sources are written in the report

### Prerequisites ###

To use this code you need:

+ TensorFlow 2.0 or higher
+ Python 3.5-3.7

### Folder strucutre and content  ###

+ "cluster": in this folder is an approch to cluster images with an autoencoder, for clarity the code is split in two files. In functions.py are the functions and imports which are needed, in main.py is the main code. Through the save weights the models must not be fit and only load and compile. Moreover, the figure 6 "Vergleich echte Klasse und vermutetet Klasse CIFAR-100" from page 8 of the report is in a hig resolution in this folder.  
+ "data": in this folder are information about the dataset. The file classnames.py print the 100 classnames with their numbers, this list is also in the file classnames with classnumbers.txt. The file unpickle.py works with the data and create the images of the subfolder data/images/. To use unpickle.py you need to download and decompress 'CIFAR-100 python version' from https://www.cs.toronto.edu/~kriz/cifar.html . In images/classes there are for all 100 class five example images. In images/superclasses there are for all 20 superclass one examaple image of the classes which are grouped in this superclass.
+ "nearest_neigbours": in this folder the nearest neigbours are searched. In the subfolder autoencoder the dimenions of the images are reduce with the autoencoder. For clarity the code is split in two files, in functions.py are the functions and imports which are needed, in main.py is the main code. Through the save weights the models must not be fit and only load and compile. The current code search the nearest neigbours for a random image, to search for a specific image, please change the line 36 "n = randrange(50000)" to the id of the specific image. In the subfolder pca are the same, but here the code is in only one file. In the subfolder comparison are the files which are needed to create the table "Comparison nearest neighbors" from the report. analyse.xlsx is the excel file of the table.
+ "reconstruction_loss": in this folder are the code which are needed to create the table "Comparison reconstruction loss" from the report. analyse.xlsx is the excel file of the table.


### Author ###

+ Ahmet Efe, email: im224@stud.uni-heidelberg.de

31.01.2021 Heidelberg, Germany
