# @Author: aditya
# @Date:   2018-02-08T16:45:14+05:30
# @Last modified by:   aditya
# @Last modified time: 2018-02-08T16:45:15+05:30



import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils1 import load_dataset, random_mini_batches, convert_to_one_hot, predict
import patch_trial
import time



def one_hot_matrix(labels, C):

    C = tf.constant(C, name = 'C')
    one_hot_matrix = tf.one_hot(labels, C, axis = 0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot


def create_placeholders(n_x, n_y):

    X = tf.placeholder(tf.float32, shape = (n_x,None))
    Y = tf.placeholder(tf.float32, shape = (n_y,None))

    return X, Y


def initialize_parameters():

    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [10,4], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [10,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [10,10], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [10,1], initializer = tf.zeros_initializer())
    #W3 = tf.get_variable("W3", [50,50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    #b3 = tf.get_variable("b3", [50,1], initializer = tf.zeros_initializer())
    W4 = tf.get_variable("W4", [1,10], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b4 = tf.get_variable("b4", [1,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,}

    return parameters


def forward_propagation(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']

    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = tf.sigmoid(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.sigmoid(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)
    A3 = tf.sigmoid(Z3)
    Z4 = tf.add(tf.matmul(W4,A3),b4)

    return Z4


def compute_cost(Z3, Y):

    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.square(logits-labels))

    return cost


def regu_cost(parameters,lambd):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']

    l2_loss = lambd * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4))

    return l2_loss




def model(X_train, Y_train, X_test, Y_test, regu,learning_rate = 0.0001, beta=0.9,
          num_epochs = 5000, minibatch_size = 64, print_cost = True):

    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    testcostslist = []

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    if(regu == 0):
        cost = compute_cost(Z3, Y)
    else:
        cost = compute_cost(Z3, Y) + regu_cost(parameters,0.001)
    optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum=beta).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        for epoch in range(num_epochs):
            t1 = time.time()
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            testcostsn = sess.run([cost], feed_dict={X: X_test, Y: Y_test})

            if print_cost == True and epoch % 100 == 0:
                t2 = time.time()
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                print ("Time: "+str(t2-t1))
                #testcosts = compute_cost(forward_propagation(X_test,parameters),Y_test)
                #testcostsn = testcosts.eval()
                print ("Test Cost: "+str(testcostsn[0]))
            outl    print("")

            if print_cost == True and epoch % 4== 0 and epoch > 100 :
                costs.append(epoch_cost)
                #testcosts = compute_cost(forward_propagation(X_test,parameters),Y_test)
                #testcostsn = testcosts.eval()
                #if testcostsn[0] < 1:
                testcostslist.append(testcostsn[0])
                #else:
                   # testcostslist.append(1.0)

        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        print(len(costs))
        print(len(testcostslist))
        plt.plot(np.squeeze(testcostslist),label = 'Test')
        plt.plot(np.squeeze(costs), label = 'Training')
        plt.ylabel('cost')
        plt.xlabel('Number of epochs (per 4 epochs) ')
        plt.title("Augmenting with noise with Learning rate =" + str(learning_rate))
        plt.legend()
        plt.savefig('noise_'+str(regu)+'.png')   # save the figure to file
        plt.close()


        return parameters


kk = patch_trial.get_data()
k = patch_trial.final_data(kk)

print np.transpose(k['train']['data']).shape
print np.transpose(k['train']['labels']).shape
print np.transpose(k['test']['data']).shape
print np.transpose(k['test']['labels']).shape
#parameters1 = model(np.transpose(k['train']['data']), np.transpose(k['train']['labels']),np.transpose(k['test']['data']), np.transpose(k['test']['labels']),0)
parameters2 = model(np.transpose(k['train']['data']), np.transpose(k['train']['labels']),np.transpose(k['test']['data']), np.transpose(k['test']['labels']),1)
