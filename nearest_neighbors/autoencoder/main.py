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

#Implement Convolutional AE
encoder_ae, model_ae = general_ae()
model_ae.load_weights('ae.weigths.h5')
model_ae.compile(SGD(1e-3, 0.9), loss=loss_function)
model_ae.summary()
"""history = model_ae.fit(x_train, x_train, batch_size=8, epochs=50, verbose=1, 
                         validation_data = (x_test,x_test), shuffle=True)
model_ae.save_weights('ae.weigths.h5')

#plot loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()"""

codes1 = encoder_ae.predict(x[:30000])
codes2 = encoder_ae.predict(x[30000:])
codes = np.concatenate((codes1,codes2))

size_of_middle_layer = 128
n = randrange(50000)
print(n)
query = x_train[n]
label = y_train[n]
query_code = encoder_ae.predict(query.reshape(1,32,32,3))

n_neigh = 6
codes = codes.reshape(-1, size_of_middle_layer*8*8)
query_code = query_code.reshape(1, size_of_middle_layer*8*8)
nbrs = NearestNeighbors(n_neighbors=n_neigh).fit(codes)
distances, indices = nbrs.kneighbors(np.array(query_code))
n_label_names = [label_names[y_train[i]] for i in indices]
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