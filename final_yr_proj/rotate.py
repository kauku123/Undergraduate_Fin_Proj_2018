from scipy.io import loadmat
import numpy as np
import tensorflow as tf


def get_data():
    data_dic = {}
    data = loadmat("Indian_pines_corrected.mat")['indian_pines_corrected']
    target_mat = loadmat("Indian_pines_gt.mat")['indian_pines_gt']
    target_mat = np.array(target_mat)
    labels = []
    """
    for i in range(145):
        for j in range(145):
            labels.append(target_mat[i , j])
    labels = np.array(labels)
    """
    #print max(labels), min(labels)
    #labels = target_mat #keras.utils.to_categorical(labels)
    #labels = np.reshape(target_mat, (21025,1))
    #print labels.shape
    #d = data
    data = np.array(data)
    data_dic['data0'] = data
    #data_dic['labels0'] = labels
    return data_dic

def rotate_image(data_dic):
    
    data_np = data_dic["data0"]
    #labels_np = data_dic["labels"]

    data_tf = tf.convert_to_tensor(data_np, np.float32)
    #labels_tf = tf.convert_to_tensor(labels_np, np.float32)
    

    #data_np = np.asarray([[[1,2],[3,4]],[[5,6],[7,8]]],np.float32)

    #data_tf = tf.convert_to_tensor(data_np,np.float32)

    sess = tf.InteractiveSession()

    for i in range(0,12):

        tf_rotate = tf.contrib.image.rotate(data_tf,np.pi/2,'BILINEAR',"something")
        rotated = tf_rotate.eval();
        s = "data"+str((i+1)*90)
        data_dic[s] = rotated
        data_tf = tf.convert_to_tensor(rotated,np.float32)
        

        
    
    sess.close()
    return data_dic

    
dictionary = get_data()
rotated_dic = rotate_image(dictionary)

