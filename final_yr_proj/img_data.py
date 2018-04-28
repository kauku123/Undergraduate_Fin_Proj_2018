import scipy.io
import numpy as np
from random import shuffle
import random
import spectral
import scipy.ndimage
from skimage.transform import rotate
import os

patch_size = 5

#DATA_PATH = os.path.join(os.getcwd(),"Data")
input_mat = scipy.io.loadmat("Indian_pines.mat")
input_mat = scipy.io.loadmat("Indian_pines.mat")['indian_pines']
target_mat = scipy.io.loadmat("Indian_pines_gt.mat")
target_mat = scipy.io.loadmat("Indian_pines_gt.mat")['indian_pines_gt']

HEIGHT = input_mat.shape[0]
WIDTH = input_mat.shape[1]
BAND = input_mat.shape[2]
PATCH_SIZE = patch_size
TRAIN_PATCH,TRAIN_LABELS,TEST_PATCH,TEST_LABELS = [],[],[],[]
CLASSES = []
COUNT = 200 #Number of patches of each class
OUTPUT_CLASSES = 16
TEST_FRAC = 0.25 #Fraction of data to be used for testing

input_mat = input_mat.astype(float)
input_mat -= np.min(input_mat)
input_mat /= np.max(input_mat)

MEAN_ARRAY = np.ndarray(shape=(BAND,),dtype=float)
for i in range(BAND):
    MEAN_ARRAY[i] = np.mean(input_mat[:,:,i])

def Patch(height_index,width_index):
    """
    Returns a mean-normalized patch, the top left corner of which
    is at (height_index, width_index)

    Inputs:
    height_index - row index of the top left corner of the image patch
    width_index - column index of the top left corner of the image patch

    Outputs:
    mean_normalized_patch - mean normalized patch of size (PATCH_SIZE, PATCH_SIZE)
    whose top left corner is at (height_index, width_index)
    """
    transpose_array = np.transpose(input_mat,(2,0,1))
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = transpose_array[:, height_slice, width_slice]
    mean_normalized_patch = []
    for i in range(patch.shape[0]):
        mean_normalized_patch.append(patch[i] - MEAN_ARRAY[i])

    return np.array(mean_normalized_patch)

for i in range(OUTPUT_CLASSES):
    CLASSES.append([])
for i in range(HEIGHT - PATCH_SIZE + 1):
    for j in range(WIDTH - PATCH_SIZE + 1):
        curr_inp = Patch(i,j)
        curr_tar = target_mat[i + int((PATCH_SIZE - 1)/2), j + int((PATCH_SIZE - 1)/2)]
        if(curr_tar!=0): #Ignore patches with unkno
            CLASSES[curr_tar-1].append(curr_inp)
for c  in CLASSES:
    print len(c)

for c in range(OUTPUT_CLASSES): #for each class
    class_population = len(CLASSES[c])
    test_split_size = int(class_population*TEST_FRAC)

    patches_of_current_class = CLASSES[c]
    shuffle(patches_of_current_class)

    #Make training and test splits
    TRAIN_PATCH.append(patches_of_current_class[:-test_split_size])

    TEST_PATCH.extend(patches_of_current_class[-test_split_size:])
    TEST_LABELS.extend(np.full(test_split_size, c, dtype=int))

for i in range(OUTPUT_CLASSES):
    if(len(TRAIN_PATCH[i])<COUNT):
        tmp = TRAIN_PATCH[i]
        for j in range(COUNT/len(TRAIN_PATCH[i])):
            shuffle(TRAIN_PATCH[i])
            TRAIN_PATCH[i] = TRAIN_PATCH[i] + tmp
    shuffle(TRAIN_PATCH[i])
    TRAIN_PATCH[i] = TRAIN_PATCH[i][:COUNT]
TRAIN_PATCH = np.asarray(TRAIN_PATCH)
TRAIN_PATCH = TRAIN_PATCH.reshape((-1,220,PATCH_SIZE,PATCH_SIZE))
TRAIN_LABELS = np.array([])
for l in range(OUTPUT_CLASSES):
    TRAIN_LABELS = np.append(TRAIN_LABELS, np.full(COUNT, l, dtype=int))
print len(TEST_PATCH)
print len(TRAIN_PATCH)
for i in range(len(TRAIN_PATCH)/(COUNT*2)):
    train_dict = {}
    start = i * (COUNT*2)
    end = (i+1) * (COUNT*2)
    file_name = 'Train_'+str(PATCH_SIZE)+'_'+str(i+1)+'.mat'
    train_dict["train_patch"] = TRAIN_PATCH[start:end]
    train_dict["train_labels"] = TRAIN_LABELS[start:end]
    scipy.io.savemat( file_name,train_dict)
    print i,

for i in range(len(TEST_PATCH)/(COUNT*2)):
    test_dict = {}
    start = i * (COUNT*2)
    end = (i+1) * (COUNT*2)
    file_name = 'Test_'+str(PATCH_SIZE)+'_'+str(i+1)+'.mat'
    test_dict["test_patch"] = TEST_PATCH[start:end]
    test_dict["test_labels"] = TEST_LABELS[start:end]
    scipy.io.savemat(file_name,test_dict)
