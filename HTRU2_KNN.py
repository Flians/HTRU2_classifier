# -- coding: utf-8 --

import tensorflow as tf
import numpy as np
import pandas
import sys

### Task1: load and process data
def process_data(train_ratio = 0.8, path = './pythonWork/HTRU2_classifier/HTRU_2.csv'):
    ## load data
    data = pandas.read_csv(path, header=None)
    #print(data.loc[:,[0,1,2,3,4,5,6,7]])
    #print(data.shape)

    x_vals = np.array([data.loc[indexs].values[0:8] for indexs in data.index])
    y_vals = np.array([y for y in data[8]])

    ## separate data into training and testing
    train_indices = np.random.choice(len(x_vals), round(len(x_vals)*train_ratio), replace=False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    x_vals_train = x_vals[train_indices]
    x_vals_test = x_vals[test_indices]
    y_vals_train = y_vals[train_indices]
    y_vals_test = y_vals[test_indices]

    return x_vals_train, x_vals_test, y_vals_train, y_vals_test

### Task2: load and choice train data in proportion
def choice_data(pos_ratio = 0.5, train_ratio = 0.15, \
    pos_path = './pythonWork/HTRU2_classifier/positive.csv', neg_path = './pythonWork/HTRU2_classifier/negative.csv'):
    ## load data
    pos_data = pandas.read_csv(pos_path, header=None)
    neg_data = pandas.read_csv(neg_path, header=None)

    x_vals_pos = np.array([pos_data.loc[indexs].values[0:8] for indexs in pos_data.index])
    y_vals_pos = np.array([y for y in pos_data[8]])

    x_vals_neg = np.array([neg_data.loc[indexs].values[0:8] for indexs in neg_data.index])
    y_vals_neg = np.array([y for y in neg_data[8]])

    sum_data = len(x_vals_pos) + len(x_vals_neg)

    if train_ratio*sum_data*pos_ratio > len(x_vals_pos):
        print("\nError: train_ratio and pos_ratio are irrational! please reset!")
        return -1

    ## separate positive data into training and testing
    train_indices_pos = np.random.choice(len(x_vals_pos), round(sum_data*train_ratio*pos_ratio), replace=False)
    test_indices_pos = np.array(list(set(range(len(x_vals_pos))) - set(train_indices_pos)))

    x_vals_train_pos = x_vals_pos[train_indices_pos]
    x_vals_test_pos = x_vals_pos[test_indices_pos]
    y_vals_train_pos = y_vals_pos[train_indices_pos]
    y_vals_test_pos = y_vals_pos[test_indices_pos]

    ## separate negative data into training and testing
    train_indices_neg = np.random.choice(len(x_vals_neg), round(sum_data*train_ratio*(1-pos_ratio)), replace=False)
    test_indices_neg = np.array(list(set(range(len(x_vals_neg))) - set(train_indices_neg)))

    x_vals_train_neg = x_vals_neg[train_indices_neg]
    x_vals_test_neg = x_vals_neg[test_indices_neg]
    y_vals_train_neg = y_vals_neg[train_indices_neg]
    y_vals_test_neg = y_vals_neg[test_indices_neg]

    ## merge positive data and negatice data
    x_vals_train = np.r_[x_vals_train_pos, x_vals_train_neg]
    x_vals_test = np.r_[x_vals_test_pos, x_vals_test_neg]
    y_vals_train = np.r_[y_vals_train_pos, y_vals_train_neg]
    y_vals_test = np.r_[y_vals_test_pos, y_vals_test_neg]

    return x_vals_train, x_vals_test, y_vals_train, y_vals_test

### train and test data
def htru2_classifier(pos_ratio = 0.5, train_ratio = 0.15, op_type = 1):
    # init feeding
    x_train = tf.placeholder(shape=[None, 8], dtype=tf.float32, name='x-train')
    x_test = tf.placeholder(shape=[8], dtype=tf.float32, name='x-test')

    # Nearest Neighbor calculation using L1 Distance
    # Calculate L1 Distance
    distance = tf.reduce_sum(tf.abs(tf.add(x_train, tf.negative(x_test))), reduction_indices=1)
    #distance = tf.sqrt(tf.reduce_sum(tf.square(tf.add(x_train, tf.negative(x_test))), reduction_indices=1))
    # Prediction: Get min distance index (Nearest neighbor)
    pred = tf.argmin(distance, 0)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess, open('./pythonWork/HTRU2_classifier/result.txt', 'a+') as f_write:
        sess.run(init)

        for _ in range(10):
            # init
            accuracy = 0.
            TP = 0.
            FP = 0.
            FN = 0.
            # load data
            if op_type == 1:
                f_write.write(">>>> Process Task1\n")
                f_write.write("args: train_ratio = " + str(train_ratio) + "\n")
                x_vals_train, x_vals_test, y_vals_train, y_vals_test = process_data(train_ratio)
            else:
                f_write.write(">>>> Process Task2\n")
                f_write.write("args: pos_ratio = " + str(pos_ratio) + ", train_ratio = " + str(train_ratio) + "\n")
                x_vals_train, x_vals_test, y_vals_train, y_vals_test = choice_data(pos_ratio, train_ratio) \
                if choice_data(pos_ratio, train_ratio) != -1 \
                    else f_write.write("\nError: train_ratio and pos_ratio are irrational! please reset!\n") and exit()
            # loop over test data
            for i in range(len(x_vals_test)):
                # Get nearest neighbor
                nn_index = sess.run(pred, feed_dict={x_train: x_vals_train, x_test: x_vals_test[i, :]})
                # Get nearest neighbor class label and compare it to its true label
                #print("Test", i, "Prediction:", y_vals_train[nn_index], "True Class:", y_vals_test[i])
                # Calculate accuracy
                if y_vals_train[nn_index] == y_vals_test[i]:
                    accuracy += 1.
                TP += 1. if y_vals_train[nn_index] == y_vals_test[i] and y_vals_train[nn_index] == 1 else 0.
                FP += 1. if y_vals_train[nn_index] != y_vals_test[i] and y_vals_train[nn_index] == 1 else 0.
                FN += 1. if y_vals_train[nn_index] != y_vals_test[i] and y_vals_train[nn_index] == 0 else 0.
            P = TP/(TP+FP)
            R = TP/(TP+FN)
            f_write.write(str(_) + " ->\n")
            f_write.write(" P = TP/(TP + FP):" + str(P) + "\n")
            f_write.write(" R = TP/(TP + FN):" + str(R) + "\n")
            f_write.write(" F1 = 2.*P*R/(P + R):" + str(2.*P*R/(P + R)) + "\n")
            f_write.write(" Accuracy:" + str(accuracy/len(x_vals_test)) + "\n")
            f_write.write("\n")
            print(_,"->")
            print(" P = TP/(TP + FP):", P)
            print(" R = TP/(TP + FN):",R)
            print(" F1 = 2.*P*R/(P + R):",2.*P*R/(P + R))
            print(" Accuracy:", accuracy/len(x_vals_test))
            print()

        f_write.write("\n#################################################\n")

if __name__ == "__main__":
    """
    print(sys.argv)
    if len(sys.argv) != 2:
        print("please input op_type(1 or 2):\n\t1: Task1\n\t2: Task2\n")
    else:
        if sys.argv[1] == '1':
            print("\n>>>> Process Task1\n")
            htru2_classifier(1)
        else:
            print("\n>>>> Process Task2\n")
            htru2_classifier(0.5, 0.2, 2)
    """
    print("\n>>>> Process Task1\n")
    htru2_classifier(1)
    for index in range(7):
        print("\n>>>> Process Task2\n")
        htru2_classifier(1./(index + 2.), 0.15, 2)

