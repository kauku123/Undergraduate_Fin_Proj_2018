# Convolutional Neural Network
from scipy.io import loadmat
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
import rotate
import matplotlib.pyplot as plt
import yaml
kk = patch_trial.get_data()
kkk = patch_trial.final_data(kk)
k = patch_trial.final_data_aug(kkk)

#x_train, y_train = k['train']['data'], k['train']['labels']
#x_val, y_val = k['val']['data'], k['val']['labels']
#x_test, y_test = k['test']['data'], k['test']['labels']
pred_img_data = kk['data']
print "**********************",pred_img_data.shape
#print x_train.shape, y_train.shape, x_test.shape, y_test.shape
'''
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
'''
model_loaded = load_model('model_CNN_old.h5')
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
#	preds = np.reshape(preds, (145,145,))	
	return preds
#prediction = model_predict(model_loaded, pred_img_data)
#print prediction.shape, prediction[0][0]
		
#print model.summary()
def get_data_1(data,target_mat):
    data_dic = {}
    #data = loadmat("Indian_pines_corrected.mat")['indian_pines_corrected']
    #target_mat = scipy.io.loadmat("Indian_pines_gt.mat")['indian_pines_gt']
    target_mat = np.array(target_mat)
    labels = []
    for i in range(145):
        for j in range(145):
            labels.append(target_mat[i , j])
    labels = np.array(labels)
    #print max(labels), min(labels)
    #labels = target_mat #keras.utils.to_categorical(labels)
    #labels = np.reshape(target_mat, (21025,1))
    #print labels.shape
    d = data
    #d1 = np.pad(d, ((2,2), (2,2), (0,0)), mode='constant', constant_values=0)
    #print d1.shape, d1
    d= np.array(d)
    d = d.astype(float)
    d -= np.min(d)
    d /= np.max(d)

    y = []
    for i in range(d.shape[2]):
         dd = np.pad(d[0:d.shape[0],0:d.shape[1],i], [(2,2),(2,2)], mode='constant')
         y.append(dd)
    y = np.array(y)

    #print y[0]
    #d_p1 = np.dstack((y))
    #print  y.shape
    y1 = []
    for i in range(2, y.shape[1]-2):
        for j in range(2, y.shape[2]-2):
            y1.append(y[:, i-2:i+3, j-2:j+3])
    yy = np.array(y1)
    y1 = np.array(y1)
    #print y1.shape,y1[0,:,2,2]
    y1 = np.transpose(y1, (0,2,3,1))
    #print y1.shape, yy[0,:,2,2] == y1[0,2,2,:]

    data = y1
    #print data.shape
    data_dic['data'] = data
    data_dic['labels'] = labels
    #print labels.shape
    #y_train = keras.utils.to_categorical(y_train)
    return data_dic


d1 = {'data0':loadmat("Indian_pines_corrected.mat")['indian_pines_corrected']}
d2 = {'data0':loadmat("Indian_pines_gt.mat")['indian_pines_gt']}
def get_LSTM_inputs(time_steps, model):
    lstm_inputs = {}
    cnn_inputs = {}
    for i in range(6,9):
	print i 
        inp = {'data0':d1['data0']}
        targ = {'data0':d2['data0']}
        for j in range(i):
            inp = rotate.rotate_image(inp)
            targ = rotate.rotate_image(targ)

        data_to_cnn = get_data_1(inp['data0'],targ['data0'])
	data_to_cnn_new = patch_trial.final_data(data_to_cnn)
	cnn_inputs[i+1] = []
	for item in data_to_cnn_new['train']['data']:
		cnn_inputs[i+1].append(item)
	for item in data_to_cnn_new['test']['data']:
		cnn_inputs[i+1].append(item)
	cnn_inputs[i+1] = np.array(cnn_inputs[i+1])
        
	#print data_to_cnn_new['train']['data'].shape,data_to_cnn_new['test']['data'].shape,temp.shape
        lstm_inputs[i+1] = model_predict(model,cnn_inputs[i+1])
    return lstm_inputs

'''inp_lstm = get_LSTM_inputs(5, model_loaded)
#print inp_lstm
with open('decoder_data.yml', 'w') as outfile:
    yaml.dump(inp_lstm, outfile, default_flow_style=False)'''
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


#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#model.fit(training_data, training_labels)
#argmax layer

#argmax_layer = K.argmax(conv_1x1_8, axis=-1)'''
