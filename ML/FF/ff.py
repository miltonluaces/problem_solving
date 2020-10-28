import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

def ForwardStep(X, model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    n1 = X.dot(W1) + b1
    a1 = np.tanh(n1)
    n2 = a1.dot(W2) + b2
    y = np.exp(n2)
    py = y / np.sum(y, axis=1, keepdims=True)
    return py, a1

def BackwardStep(X, y, nExamples, py, a1, model, m, Lambda):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Output to hidden layer
    d2 = py
    d2[range(nExamples), y] -= 1
    dW2 = (a1.T).dot(d2)
    db2 = np.sum(d2, axis=0, keepdims=True)
 
    # Hidden layer
    d1 = d2.dot(W2.T) * (1 - np.power(a1, 2))
    dW1 = np.dot(X.T, d1)
    db1 = np.sum(d1, axis=0)

    # Add regularization terms (only Ws)
    dW1 += Lambda * W1
    dW2 += Lambda * W2
  
    # Gradient descent parameter update
    W1 -= m * dW1
    b1 -= m * db1
    W2 -= m * dW2
    b2 -= m * db2
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return(model)

def CalcLoss(model, X, y, Lambda):
    nExamples = len(X) 
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    py, a1 = ForwardStep(X, model)
    
    # Calculating the loss
    logPy = -np.log(py[range(nExamples), y])
    loss = np.sum(logPy)
    
    # Add regulatization to loss
    loss += Lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / (nExamples * loss)

def BuildModel(X, y, nInp, nHid, nOut, epochs, m, Lambda, printLoss=False):
    
    # Initialize 
    nExamples = len(X)
    np.random.seed(0)
    W1 = np.random.randn(nInp, nHid) / np.sqrt(nInp)
    b1 = np.zeros((1, nHid))
    W2 = np.random.randn(nHid, nOut) / np.sqrt(nHid)
    b2 = np.zeros((1, nOut))
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
     
    # Train
    for i in range(0, epochs):

        py, a1 = ForwardStep(X, model)
        model = BackwardStep(X, y, nExamples, py, a1, model, m, Lambda)
        
        if printLoss and i % 1000 == 0: print("Loss after iteration %i: %f" % (i, CalcLoss(model, X, y, Lambda)))

    return model

def Predict(model, X):
    py, a1 = ForwardStep(X, model)
    return np.argmax(py, axis=1)

def plot_decision_boundary(predFunc, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = predFunc(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

# Generate data
X, y = datasets.make_moons(200, noise=0.20)

# Logistic Regression
lr = linear_model.LogisticRegressionCV()
lr.fit(X, y)
plot_decision_boundary(lambda x: lr.predict(x), X, y)
plt.title("Logistic Regression")
plt.show()

# Neural Networks
nn = BuildModel(X=X, y=y, nInp=2, nHid=3, nOut=2, epochs=20000, m=0.01, Lambda=0.01, printLoss=True)
plot_decision_boundary(lambda x:Predict(nn,x), X, y)
plt.title("Neural Network")
plt.show()