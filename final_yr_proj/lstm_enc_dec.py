from scipy.io import loadmat
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
import patch_trial
import rotate
import matplotlib.pyplot as plt
import ker_fnc
import yaml
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
input_2_enc = np.array(input_2_enc)
target_2_dec = np.array(target_2_dec)
input_2_dec = np.array(input_2_dec)
input_2_enc = np.reshape(input_2_enc, (1,input_2_enc.shape[0], input_2_enc.shape[1]))
input_2_dec = np.reshape(input_2_dec, (1,input_2_dec.shape[0], input_2_dec.shape[1]))
target_2_dec = np.reshape(target_2_dec, (1,target_2_dec.shape[0], target_2_dec.shape[1]))
print input_2_enc.shape, input_2_dec.shape, target_2_dec.shape
#Encoder Model

encoder_timesteps = 5
encoder_input = Input(shape=(input_2_enc.shape[1], input_2_enc.shape[2]))
encoder_LSTM = LSTM(256, return_state = True)
encoder_outputs, encoder_h, encoder_c = encoder_LSTM(encoder_input)
encoder_states = [encoder_h, encoder_c]

#Decoder Model
decoder_timesteps = 3
decoder_input = Input(shape=(input_2_dec.shape[1], input_2_dec.shape[2]))
decoder_LSTM = LSTM(256, return_sequences=True, return_state=True)
decoder_out, _, _ = decoder_LSTM(decoder_input, initial_state=encoder_states)
decoder_dense = Dense(target_2_dec.shape[2], activation='softmax')
decoder_out = decoder_dense(decoder_out)

model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_out])

model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.fit(x=[input_2_enc, input_2_dec], y = target_2_dec, batch_size=8, epochs=500, verbose=2)


