# -- coding: utf-8 --
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas
from sklearn import datasets

## load data
data = pandas.read_csv('./pythonWork/HTRU2_ classifier/HTRU_2.csv',header=None)
#print(data.loc[:,[0,1,2,3,4,5,6,7]])
print(data.shape)

x_vals = np.array([data.loc[indexs].values[0:8] for indexs in data.index])
y_vals = np.array([1 if y==1 else -1 for y in data[8]])

## separate data into training and testing
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

## define Model and Loss
batch_size = 50

# init feeding
x_data = tf.placeholder(shape=[None, 8], dtype=tf.float32, name='x-input')
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='y-output')

# create Variable
A = tf.Variable(tf.random_normal(shape=[8, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# define Linear model y = Ax + b
model_output = tf.add(tf.matmul(x_data, A), b)

# Declare vector L2 'norm' function squared
l2_norm = tf.reduce_sum(tf.square(A))

# Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
alpha = tf.constant([0.01])
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

## training and testing
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

#lasting
saver = tf.train.Saver()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training loop
    for i in range(100000):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    saver.save(sess, "./pythonWork/HTRU2_ classifier/model.ckpt")
    print(A.eval(session=sess))
    print(b.eval(session=sess))

    # testing
    result = tf.maximum(0.,tf.multiply(model_output, y_target))
    #saver.restore(sess, "./pythonWork/HTRU2_ classifier/model.ckpt")
    y_test = np.reshape(y_vals_test, (len(y_vals_test),1))
    array = sess.run(result, feed_dict={x_data: x_vals_test, y_target: y_test})
    num = np.array(array)
    zero_num = np.sum(num==[0.])
    print(num)
    print(zero_num)
    print("Actually:",zero_num/len(num))
