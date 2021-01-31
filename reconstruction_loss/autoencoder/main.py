from functions import *

#load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
label_names = get_label_names()

#data normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.
x_test = x_test / 255.
x = np.concatenate((x_train,x_test))
y = np.concatenate((y_train,y_test))

#Implement Convolutional AE
encoder_ae, model_ae = general_ae()
model_ae.load_weights('ae.weigths.h5')
model_ae.compile(SGD(1e-3, 0.9), loss=loss_function)
model_ae.summary()
"""history = model_ae.fit(x_train, x_train, batch_size=8, epochs=100, verbose=1, 
                         validation_data = (x_test,x_test), shuffle=True)
model_ae.save_weights('ae.weigths.h5')

#plot loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()"""

output_image1 = model_ae.predict(x[:30000])
output_image2 = model_ae.predict(x[30000:])
output_image = np.concatenate((output_image1, output_image2))

list_of_norm = []
for i in range(len(x)):
    diff = x[i] - output_image[i]
    norm = sum(sum(sum(abs(diff))))
    list_of_norm.append(norm)

with open('pca_list_of_norm.txt', 'w') as f:
    f.write(json.dumps(list_of_norm))
