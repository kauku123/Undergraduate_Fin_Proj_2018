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


#x_train =
'''x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)'''
input_shape = (5,5,220)
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))
'''model.add(Conv2D(8, (3, 3), input_shape=input_shape, padding='same',
       activation='relu'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(AveragePooling2D(pool_size=(2, 2)))'''


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(x_train,y_train, batch_size=32, epochs=20,validation_split=0.1,verbose=5)#,validation_data=(x_val,y_val))
#score = model.evaluate(x_test, y_test, batch_size=32)
#clf = KerasClassifier(build_fn=baseline_model, epochs=20000, batch_size=128,verbose=2)
#kfold = KFold(n_splits=10,shuffle=True,random_state=seed)
#results = cross_val_score(clf, x_train, y_train, cv=kfold)
'''model.fit(x_train, y_train,
          epochs=500, batch_size=128,validation_data=(x_test, y_test))'''
score = model.evaluate(x_test, y_test)
print "********************Accuracy***************************" + "\n" + str(score)
'''
from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=1, decision_function_shape='ovo')
clf.fit(x_train,y_train)
clf.predict(x_test)
print clf.score(x_test, y_test)



for i in data:
	if '__' not in i and 'readme' not in i:
		np.savetxt((i+".csv"),data[i],delimiter=',')'''
