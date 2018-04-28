# Convolutional Neural Network
import numpy as np
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Flatten,Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.layers.merge import concatenate
from keras.layers.merge import add
from keras.optimizers import SGD, Adam
import keras.layers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
#from keras.layers.merge import
import patch_trial
import matplotlib.pyplot as plt
kk = patch_trial.get_data()
kkk = patch_trial.final_data(kk)
k = patch_trial.final_data_aug(kkk)

x_train, y_train = k['train']['data'], k['train']['labels']
#x_val, y_val = k['val']['data'], k['val']['labels']
x_test, y_test = k['test']['data'], k['test']['labels']
pred_img_data = kk['data']
print x_train.shape, y_train.shape, x_test.shape, y_test.shape

H, W = 5, 5
#input Layer:

inp = Input(shape=(H,W,200))
dropout1 = Dropout(0.5)
dropout2 = Dropout(0.5)
optimizr = Adam(lr = 0.0001,  beta_1=0.9, beta_2=0.999, epsilon=1e-8)

#5x5 conv filter
conv_5x5 = Conv2D(128, kernel_size=(5,5), activation='relu')(inp)

#3x3 conv filter
conv_3x3 = Conv2D(128, kernel_size=(3,3), activation='relu')(inp)
pool_3x3 = MaxPooling2D(pool_size=(3,3),strides=1)(conv_3x3)

#1x1 conv filter
conv_1x1 = Conv2D(128, kernel_size=(1,1), activation='relu')(inp)
pool_1x1 = MaxPooling2D(pool_size=(5,5),strides=1)(conv_1x1)

#Multi_Scale_Filter_Bank
first_concat_output = concatenate([conv_5x5, pool_3x3, pool_1x1])
norm_layer = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(first_concat_output)
#First_Residual Unit
conv_1x1_1 = Conv2D(128, kernel_size=(1,1),activation='relu')(norm_layer)
norm_layer1 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(conv_1x1_1)
conv_1x1_2 = Conv2D(128, kernel_size=(1,1), activation='relu')(norm_layer1)
conv_1x1_3 = Conv2D(128, kernel_size=(1,1), activation='relu')(conv_1x1_2)
first_ReLU_sum = add([conv_1x1_1, conv_1x1_3])



#Seconf_Residual Unit
conv_1x1_4 = Conv2D(128, kernel_size=(1,1),activation='relu')(first_ReLU_sum)
conv_1x1_5 = Conv2D(128, kernel_size=(1,1), activation='relu')(conv_1x1_4)
second_ReLU_sum = add([first_ReLU_sum, conv_1x1_5])

conv_1x1_6 = Conv2D(128, kernel_size=(1,1),activation='relu')(second_ReLU_sum)
dropped_conv_1x1_6 = dropout1(conv_1x1_6)
conv_1x1_7 = Conv2D(128, kernel_size=(1,1),activation='relu')(dropped_conv_1x1_6)
dropped_conv_1x1_7 = dropout2(conv_1x1_7)
conv_1x1_8 = Conv2D(128, kernel_size=(1,1),activation='relu')(dropped_conv_1x1_7)
flat_vector = Flatten()(conv_1x1_8)
fcn1 = Dense(64, activation='relu')(flat_vector)

fcn = Dense(8, activation='softmax')(flat_vector)
model = Model(inputs=inp, outputs=fcn)
model.compile(optimizer=optimizr,loss='categorical_crossentropy',metrics=['accuracy'])
#checkpoint = ModelCheckpoint( 'model_CNN.h5' , monitor= 'val_loss' , verbose=0,
#save_best_only=True, mode= 'min' )
#history = model.fit(x_train, y_train, epochs = 600, batch_size=16, shuffle=True, verbose=2, validation_split=0.1, callbacks=[checkpoint])#, validation_data=(x_val, y_val))
#score = model.evaluate(x_test, y_test, batch_size=32)
#print score
model_loaded = load_model('model_CNN.h5')
preds = []
def model_predict(model, data):
	preds = []
	predicts = model.predict(data, batch_size=data.shape[0])#, batch_size=53336, verbose=2)
	print "Predicts"
	#print type(predicts), predicts.shape, predicts[0]
	for i in range(predicts.shape[0]):
		preds.append(np.argmax(predicts[i]))
		#print preds[i]
	preds = np.array(preds)
	preds = np.reshape(preds, (145,145,))	
	return preds
prediction = model_predict(model_loaded, pred_img_data)
print prediction.shape, prediction[0][0]
		
#print model.summary()
'''plot_model(model, to_file='dcn.png')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''

'''
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(training_data, training_labels)
#argmax layer

#argmax_layer = K.argmax(conv_1x1_8, axis=-1)'''
