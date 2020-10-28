# Imports
#region 

import numpy as np
import scipy as sp
import math
import tensorflow as tf
import matplotlib as mp
import kiwisolver as kw
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd

#endregion


def mpTest():
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2*np.pi*t)
    plt.plot(t, s)

    plt.xlabel('time (s)')
    plt.ylabel('voltage (mV)')
    plt.title('About as simple as it gets, folks')
    plt.grid(True)
    plt.savefig("test.png")
    plt.show()

def skTest():
    from sklearn import datasets, linear_model
    from sklearn.metrics import mean_squared_error, r2_score
    diabetes = datasets.load_diabetes()
    diabetes_X = diabetes.data[:, np.newaxis, 2]
    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)
    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
    plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def TfInstallation():
    sess = tf.Session()

    # Verify we can print a string
    hello = tf.constant("Hello Pluralsight from TensorFlow")
    print(sess.run(hello))

    #   Perform some simple math
    a = tf.constant(20)
    b = tf.constant(22)
    print('a + b = {0}'.format(sess.run(a + b)))

def TensorboardTest():

    sess = tf.Session()

    a = tf.constant(3, name="a")
    b = tf.constant(2, name="b")
    s = tf.add_n([a,b], name="sum")
    print(sess.run(s))

    writer = tf.summary.FileWriter('.')
    writer.add_graph(tf.get_default_graph())
    


