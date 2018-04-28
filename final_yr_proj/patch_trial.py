from scipy.io import loadmat
import numpy as np
import scipy.io
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
import pandas as pd
from keras import backend as K
from keras.utils import to_categorical
import tensorflow as tf


def get_data():
    data_dic = {}
    data = loadmat("Indian_pines_corrected.mat")['indian_pines_corrected']
    target_mat = scipy.io.loadmat("Indian_pines_gt.mat")['indian_pines_gt']
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
k = get_data()
#print k
#print k
def final_data(k):
    fin_data_dic = {}
    top_classes = list(pd.Series(k['labels']).value_counts().index[0:8])
    #print "TopClasses****************",top_classes
    k_df = pd.DataFrame(k['labels'], columns=['a'])
    #print y_df.values, y_df.values.shape
    k_df_loc = k_df.loc[k_df['a'].isin(top_classes),:]
    #y_df_loc1 = y_df.loc[y_df['a'],:]
    top_8_index = k_df_loc.index


    y_new = []
    X_new = []
    for i in top_8_index:
        X_new.append(k['data'][i])
        y_new.append(k['labels'][i])

    X_new = np.array(X_new)

    #y_new = np.array(y_new)
    #y_new = np.reshape(y_new, (y_new.shape[0],1))
    #print X_new.shape, len(y_new), min(y_new), max(y_new)

    #print (X_new.shape)
    count_list = np.zeros((8))
    #print count_list

    for i in range(len(y_new)):
        if y_new[i] == 0:
            count_list[0] += 1
            y_new[i]= int(0)
        elif y_new[i] == 11:
            count_list[1] += 1
            y_new[i]= int(1)
        elif y_new[i] == 2:
            count_list[2] += 1
            y_new[i] = int(2)
        elif y_new[i] == 14:
            count_list[3] += 1
            y_new[i] = int(3)
        elif y_new[i] == 10:
            count_list[4] += 1
            y_new[i] = int(4)
        elif y_new[i] == 3:
            count_list[5] += 1
            y_new[i] = int(5)
        elif y_new[i] == 6:
            count_list[6] += 1
            y_new[i]= int(6)
        elif y_new[i] == 12:
            count_list[7] += 1
            y_new[i] = int(7)

    X_new,y_new = shuffle(X_new,y_new,random_state = 0)
    rcount_list = np.round(0.7*count_list)
    #rcount_list = [200, 200, 200, 200, 200, 200, 200, 200]
    #print sum(rcount_list)
    x_train, x_test, y_train, y_test = [],[],[],[]
    for i in range(len(y_new)):
        if y_new[i] == 0:
            rcount_list[0] -= 1
            if (rcount_list[0] >= 0):
                x_train.append(X_new[i])
                y_train.append(y_new[i])
            else:
                x_test.append(X_new[i])
                y_test.append(y_new[i])
        elif y_new[i] == 1:
            rcount_list[1] -= 1
            if (rcount_list[1] >= 0):
                x_train.append(X_new[i])
                y_train.append(y_new[i])
            else:
                x_test.append(X_new[i])
                y_test.append(y_new[i])
        elif y_new[i] == 2:
            rcount_list[2] -= 1
            if (rcount_list[2] >= 0):
                x_train.append(X_new[i])
                y_train.append(y_new[i])
            else:
                x_test.append(X_new[i])
                y_test.append(y_new[i])
        elif y_new[i] == 3:
            rcount_list[3] -= 1
            if (rcount_list[3] >= 0):
                x_train.append(X_new[i])
                y_train.append(y_new[i])
            else:
                x_test.append(X_new[i])
                y_test.append(y_new[i])
        elif y_new[i] == 4:
            rcount_list[4] -= 1
            if (rcount_list[4] >= 0):
                x_train.append(X_new[i])
                y_train.append(y_new[i])
            else:
                x_test.append(X_new[i])
                y_test.append(y_new[i])
        elif y_new[i] == 5:
            rcount_list[5] -= 1
            if (rcount_list[5] >= 0):
                x_train.append(X_new[i])
                y_train.append(y_new[i])
            else:
                x_test.append(X_new[i])
                y_test.append(y_new[i])
        elif y_new[i] == 6:
            rcount_list[6] -= 1
            if (rcount_list[6] >= 0):
                x_train.append(X_new[i])
                y_train.append(y_new[i])
            else:
                x_test.append(X_new[i])
                y_test.append(y_new[i])
        elif y_new[i] == 7:
            rcount_list[7] -= 1
            if (rcount_list[7] >= 0):
                x_train.append(X_new[i])
                y_train.append(y_new[i])
            else:
                x_test.append(X_new[i])
                y_test.append(y_new[i])
    print (len(y_train),len(x_train))
    #print(len(x_train)+len(x_test))
    x_train1 = np.array(x_train)
    print x_train1.shape
    y_train1 = np.array(y_train)
    np.random.seed(1)
    #noise_trainp = np.random.normal(0,1,x_train1.shape)
    #noise_trainm = np.random.normal(0,1,x_train1.shape)
    #print noise_train.shape
    #x_trainnp = x_train1 + noise_trainp
    #x_trainnm = x_train1 - noise_trainm
    #x_train.extend(x_trainnp)
    #x_train.extend(x_trainnm)
    #y_train.extend(y_train)
    #print 'y',len(y_train)
    #y_train.extend(y_train1)
    #print len(y_train)
    x_train = np.array(x_train)
    #print x_train.shape
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    #print y_train.shape

    y_test = np.array(y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    '''
    #def one_hot_enc(l):
    y_new = np.array(y_new)
    y_new = to_categorical(y_new)
    x_train, y_train = X_new[0:int(0.75*X_new.shape[0]),:,:,:], y_new[0:int(0.75*y_new.shape[0]),:]
    #x_val, y_val = X_new[int(0.6*X_new.shape[0]):int(0.8*X_new.shape[0]),:,:,:], y_new[int(0.6*y_new.shape[0]):int(0.8*y_new.shape[0]),:]
    x_test, y_test = X_new[int(0.75*X_new.shape[0]):int(1.0*X_new.shape[0]),:,:,:], y_new[int(0.75*y_new.shape[0]):int(1.0*y_new.shape[0]),:]
    '''
    dic_train, dic_val, dic_test = {}, {}, {}


    dic_train['data'], dic_train['labels'] = x_train, y_train
    #dic_val['data'], dic_val['labels'] = x_val, y_val
    dic_test['data'], dic_test['labels'] = x_test, y_test
    fin_data_dic['train'] = dic_train
    #fin_data_dic['val'] = dic_val
    fin_data_dic['test'] = dic_test
    #print X_new.shape, y_new.shape
    return fin_data_dic#, min(y_new), max(y_new)

#kk = final_data(k)
#print kk['train']['labels'].shape

def final_data_aug(kk):
    dic_train_aug, dic_test_aug = {}, {}
    fin_data_dic_aug = {}
    x_train, x_test = kk['train']['data'], kk['test']['data']
    y_train, y_test = kk['train']['labels'], kk['test']['labels']
    print x_train.shape
    #x_train, x_test = list(x_train), list(x_test)
    #y_train, y_test = list(y_train), list(y_test)

    x_train_aug, x_test_aug, y_train_aug, y_test_aug = [], [], [], []

    for i in x_train:
        #print i.shape
        x_train_aug.append(i)
        x_train_aug.append(np.flipud(i))
        x_train_aug.append(np.fliplr(i))
        x_train_aug.append(np.transpose(i, (1,0,2)))
    tot_len = len(x_train_aug)
    x_train_aug = np.array(x_train_aug)
    #x_train_aug = np.reshape(x_train_aug, (tot_len,5,5,220))
    print x_train_aug[2].shape

    for i1 in x_test:
        #print i.shape
        x_test_aug.append(i1)
        x_test_aug.append(np.flipud(i1))
        x_test_aug.append(np.fliplr(i1))
        x_test_aug.append(np.transpose(i1, (1,0,2)))
    tot_len = len(x_test_aug)
    x_test_aug = np.array(x_test_aug)
    #x_train_aug = np.reshape(x_train_aug, (tot_len,5,5,220))
    print x_test_aug.shape


    for i2 in y_train:
        #print i.shape
        y_train_aug.append(i2)
        y_train_aug.append(i2)
        y_train_aug.append(i2)
        y_train_aug.append(i2)
    #tot_len = len(x_train_aug)
    y_train_aug = np.array(y_train_aug)
    #x_train_aug = np.reshape(x_train_aug, (tot_len,5,5,220))
    print y_train_aug.shape

    for i3 in y_test:
        #print i.shape
        y_test_aug.append(i3)
        y_test_aug.append(i3)
        y_test_aug.append(i3)
        y_test_aug.append(i3)
    #tot_len = len(x_train_aug)
    y_test_aug = np.array(y_test_aug)
    #x_test_aug = np.reshape(x_test_aug, (tot_len,5,5,220))
    print y_test_aug.shape

    dic_train_aug['data'], dic_test_aug['data'], dic_train_aug['labels'], dic_test_aug['labels'] = x_train_aug, x_test_aug, y_train_aug, y_test_aug

    fin_data_dic_aug['train'] = dic_train_aug
    fin_data_dic_aug['test'] = dic_test_aug
    return fin_data_dic_aug

#kkk = final_data_aug(kk)
#print kkk['train']['data'].shape

'''for i in top_8_index:

    X_new.append(k['data'][i])
    y_new.append(k['labels'][i])
    print X_new, y_new, i
print  len(y_new),max(y_new),min(y_new)
'''


#print y1[0]
