import scipy.io
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import pandas as pd

from skimage.util import view_as_windows as viewW


# Convolutional Neural Network
from keras.utils import plot_model
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense, Flatten,Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras import backend as K
from keras.layers.merge import concatenate
from keras.layers.merge import add
import keras.layers
#from keras.layers.merge import
import patch_trial
import matplotlib.pyplot as plt
kk = patch_trial.get_data()
k = patch_trial.final_data(kk)
x_train, y_train = k['train']['data'], k['train']['labels']
#x_val, y_val = k['val']['data'], k['val']['labels']
x_test, y_test = k['test']['data'], k['test']['labels']

seed = np.random.seed(1)

def baseline_model():
	model = Sequential()
	# Dense(64) is a fully-connected layer with 64 hidden units.
	# in the first layer, you must specify the expected input data shape:
	# here, 20-dimensional vectors.
	model.add(Dense(220, activation='relu', input_shape=(5,5,220)))
	model.add(Dropout(0.5))
	#model.add(Dense(500, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(30, activation='relu'))
	model.add(Dropout(0.5))
	#model.add(Dense(50, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(8, activation='softmax'))

	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',
	              optimizer=sgd,
	              metrics=['accuracy'])
	return model
clf = KerasClassifier(build_fn=baseline_model, epochs=20000, batch_size=128,verbose=2)
kfold = KFold(n_splits=10,shuffle=True,random_state=seed)
results = cross_val_score(clf, x_train, y_train, cv=kfold)
model.fit(x_train, y_train,
          epochs=500, batch_size=128)#,validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test,batch_size=128)
#print "********************Accuracy***************************" + "\n" + str(results.mean())
