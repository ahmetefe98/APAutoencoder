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

size_of_middle_layer = 64
codes1 = encoder_ae.predict(x[:30000])
codes2 = encoder_ae.predict(x[30000:])
codes = np.concatenate((codes1,codes2))

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
        query_code = encoder_ae.predict(query.reshape(1,32,32,3))
        n_neigh = 6
        codes = codes.reshape(-1, size_of_middle_layer*8*8)
        query_code = query_code.reshape(1, size_of_middle_layer*8*8)
        nbrs = NearestNeighbors(n_neighbors=n_neigh, n_jobs = -1).fit(codes)
        distances, indices = nbrs.kneighbors(np.array(query_code))
        n_label_names = [label_names[y[i]] for i in indices]
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
