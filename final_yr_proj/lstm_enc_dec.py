#from scipy.io import loadmat
import numpy as np
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, LSTM
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
#import patch_trial
#import rotate
#import matplotlib.pyplot as plt
#import ker_fnc
#import yaml
import tensorflow as tf
'''
with open("encoder_data.yml", 'r') as stream:
    lstm_enc_inputs = yaml.load(stream)

with open("decoder_data.yml", 'r') as stream:
    lstm_dec_inputs = yaml.load(stream)
print lstm_dec_inputs
input_2_enc = []
input_2_dec = []
target_2_dec = []
for i in range(1,6):
	input_2_enc.append(lstm_enc_inputs[i])
for i in range(6,9):
	target_2_dec.append(lstm_dec_inputs[i+1])
input_2_dec.append(input_2_enc[4])
for ii in range(len(target_2_dec)-1):
	input_2_dec.append(target_2_dec[ii])
'''

def rotate(img):
	return np.fliplr(np.transpose(img))


np.random.seed(63075)
original_data = np.random.randint(0,8,size=(20000,145,145))
'''rotation_1 = []
rotation_2 = []
rotation_3 = []
rotation_4 = []
rotation_5 = []
rotation_6 = []
rotation_7 = []

temp = [original_data,rotation_1,rotation_2,rotation_3,rotation_4,rotation_5,rotation_6,rotation_7]
for i in range(len(temp) - 1):
	for img in temp[i]:
		temp[i+1].append(rotate(img))
rotation_1 = np.array(rotation_1)
rotation_2 = np.array(rotation_2)
rotation_3 = np.array(rotation_3)
rotation_4 = np.array(rotation_4)
rotation_5 = np.array(rotation_5)
rotation_6 = np.array(rotation_6)
rotation_7 = np.array(rotation_7)'''

input_2_enc = []
target_2_dec = []
input_2_dec = []
for i in range(len(original_data)):
	print i
	rot_1 = rotate(original_data[i])
	rot_2 = rotate(rot_1)
	rot_3 = rotate(rot_2)
	vec_org = np.reshape(original_data[i],(21025))
	vec_1 = np.reshape(rot_1,(21025))	
	vec_2 =	np.reshape(rot_2,(21025))
	vec_3 = np.reshape(rot_3,(21025))
	input_2_enc.append([vec_org,vec_1,vec_2,vec_3,vec_org])
	target_2_dec.append([vec_1,vec_2,vec_3])
	input_2_dec.append([vec_org,vec_1,vec_2])
input_2_enc = np.array(input_2_enc)
target_2_dec = np.array(target_2_dec)
input_2_dec = np.array(input_2_dec)
print input_2_enc.shape
print target_2_dec.shape
print input_2_dec.shape
'''print original_data[0,0,0],original_data[0,2,0],original_data[0,0,144]
print rotation_1[0,144,0],rotation_1[0,144,2],rotation_1[0,0,0]
print rotation_2[0,144,144]
print rotation_3[0,0,144]
print original_data[0] - rotation_4[0]'''

'''input_2_enc = np.array(input_2_enc)
target_2_dec = np.array(target_2_dec)
input_2_dec = np.array(input_2_dec)
input_2_enc = np.reshape(input_2_enc, (1,input_2_enc.shape[0], input_2_enc.shape[1]))
input_2_dec = np.reshape(input_2_dec, (1,input_2_dec.shape[0], input_2_dec.shape[1]))
target_2_dec = np.reshape(target_2_dec, (1,target_2_dec.shape[0], target_2_dec.shape[1]))
print input_2_enc.shape, input_2_dec.shape, target_2_dec.shape'''

#Encoder Model

encoder_timesteps = 5
encoder_input = Input(shape=(input_2_enc.shape[1], input_2_enc.shape[2]))
encoder_LSTM = LSTM(16, return_state = True)
encoder_outputs, encoder_h, encoder_c = encoder_LSTM(encoder_input)
encoder_states = [encoder_h, encoder_c]

#Decoder Model
decoder_timesteps = 3
decoder_input = Input(shape=(input_2_dec.shape[1], input_2_dec.shape[2]))
decoder_LSTM = LSTM(16, return_sequences=True, return_state=True)
decoder_out, _, _ = decoder_LSTM(decoder_input, initial_state=encoder_states)
decoder_dense = Dense(target_2_dec.shape[2], activation='softmax')
decoder_out = decoder_dense(decoder_out)

model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_out])
model.summary()
model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.fit(x=[input_2_enc, input_2_dec], y = target_2_dec, batch_size=8, epochs=500, verbose=2)
#plot_model(model, to_file='lstm_en_dec.png')

