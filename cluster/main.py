from functions import *

#load data
print('load data')
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

sub_and_classnames = ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
                    'orchid', 'poppy', 'rose', 'sunflower', 'tulip', 'bottle', 'bowl', 'can', 'cup', 'plate',
                    'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper','clock', 'keyboard', 'lamp', 'telephone', 'television',
                     'bed', 'chair', 'couch', 'table', 'wardrobe','bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                     'bear', 'leopard', 'lion', 'tiger', 'wolf', 'bridge', 'castle', 'house', 'road', 'skyscraper',
                    'cloud', 'forest', 'mountain', 'plain', 'sea','camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                    'fox', 'porcupine', 'possum', 'raccoon', 'skunk','crab', 'lobster', 'snail', 'spider', 'worm',
                    'baby', 'boy', 'girl', 'man', 'woman','crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                    'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel','maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 
                    'willow_tree','bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train','lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']

dict = dict()
for i in range(len(sub_and_classnames)):
        dict[i] = sub_and_classnames[i]
        
#data normalization
print('data normalization')
x_train_final = x_train
x_test_final = x_test
y_train_final = y_train
y_test_final = y_test
x_train_final = x_train_final.astype('float32')
x_test_final = x_test_final.astype('float32')
x_train_final = x_train_final / 255
x_test_final = x_test_final / 255

#Split the data
print('Split the data')
x_train, x_valid, y_trainf, y_validf = train_test_split(x_train_final, y_train_final, test_size=0.2, random_state=42, shuffle= True)

#Target conversion to categorical
print('Target conversion to categorical')
y_train = keras.utils.to_categorical(y_trainf, 100)
y_valid = keras.utils.to_categorical(y_validf, 100)
y_test_one_hot = keras.utils.to_categorical(y_test_final, 100)

#Implement Convolutional AE
print('Implement Convolutional AE')
encoder_ae, model_ae = general_ae()
model_ae.load_weights('ae.weigths.h5')
model_ae.compile(SGD(1e-3, 0.9), loss=loss_function)
model_ae.summary()
"""history = model_ae.fit(x_train, x_train, 
                       batch_size=8,
                       epochs=100,
                       verbose=1,
                       validation_data=(x_valid, x_valid),
                       shuffle=True)
model_ae.save_weights('ae.weigths.h5')

print('plot')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()"""

recon_test_ae = model_ae.predict(x_test)
recon_valid_ae = model_ae.predict(x_valid)
showOrigDec(x_valid, recon_valid_ae)
showOrigDec(x_test, recon_test_ae)

#Extracting bottleneck features to use as inputs in the classifier model
gist_train_ae = encoder_ae.predict(x_train)
gist_valid_ae = encoder_ae.predict(x_valid)
gist_test_ae = encoder_ae.predict(x_test_final)

#with simple nn as classifier
decoder_ae_dense = classifier_dense(gist_train_ae)
decoder_ae_dense.load_weights('dae.weigths.h5')
decoder_ae_dense.compile(loss='categorical_crossentropy',
                         optimizer=Adadelta(),
                         metrics=['accuracy'])
decoder_ae_dense.summary()
er = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_delta=0.0001)
callbacks = [er, lr]
"""hist1 = decoder_ae_dense.fit(gist_train_ae, y_train, batch_size=8, epochs=100, 
                             validation_data = (gist_valid_ae, y_valid),
                             shuffle=True, callbacks=callbacks)
decoder_ae_dense.save_weights('dae.weigths.h5')

plt.plot(hist1.history['accuracy'])
plt.plot(hist1.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()"""

print('Test accuracy for AE_dense model= {}'.format(decoder_ae_dense.evaluate(gist_test_ae, y_test_one_hot)[1]))

show_test(decoder_ae_dense, gist_test_ae, x_test_final, y_test_final, dict)
predictions = decoder_ae_dense.predict(gist_test_ae)
report(predictions, y_test_one_hot, dict)