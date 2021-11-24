# Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
import time


def readData(path):
    tmp = pd.read_csv(path).values
    variable = preprocessing.scale(tmp)
    return variable


def load_data():
    __filepath__ = ['../data/X_train.csv', '../data/Y_train.csv', '../data/X_test.csv', '../data/Y_test.csv']
    __X_train__ = pd.read_csv(__filepath__[0]).values
    __X_train__ = preprocessing.scale(__X_train__)
    __Y_train__ = pd.read_csv(__filepath__[1])['label']
    __Y_train__ = __Y_train__.values
    __X_test__ = pd.read_csv(__filepath__[2]).values
    __X_test__ = preprocessing.scale(__X_test__)
    __Y_test__ = pd.read_csv(__filepath__[3])['label'].values
    __X_train__ = __X_train__.reshape(__X_train__.shape[0], -1).T
    __Y_train__ = __Y_train__.reshape(__Y_train__.shape[0], -1).T
    __X_test__ = __X_test__.reshape(__X_test__.shape[0], -1).T
    __Y_test__ = __Y_test__.reshape(__Y_test__.shape[0], -1).T
    return __X_train__, __Y_train__, __X_test__, __Y_test__


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def propagate(w, b, X, Y):

    # Sample number m:
    m = X.shape[1]

    # Forward propagation:
    A = sigmoid(np.dot(w.T, X) + b)  # use sigmoid func
    cost = -(np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m

    # Back propagation:
    dZ = A - Y
    dw = (np.dot(X, dZ.T)) / m
    db = (np.sum(dZ)) / m

    # return value
    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, costs, startStep, print_cost=False):
    for i in range(num_iterations):
        # use propagate func to calculate the cost and gradient every iteration
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        # update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # every num_iterations iterations，save the cost
        if i % num_iterations == 0:
            costs.append(cost)

        # print cost every num_iterations to keep track of the progress of the model
        if print_cost and i % (num_iterations / 10) == 0:
            print("Cost after iteration %i: %f" % (i+startStep, cost))
    # After iterating, place the final parameters in the dictionary and return:
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(m):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction


def logistic_model(x_train, y_train, x_test, y_test, learning_rate=0.1, num_iterations=2000, print_cost=False):
    # To obtain characteristic dimensions, initialization parameters:
    dim = X_train.shape[0]
    W, b = initialize_with_zeros(dim)
    step = 100
    # Define a Costs array to store the cost after each several iterations, so that we can draw a graph to see the change trend of cost:
    costs = []
    # Define the accuracy array to store accuracy after several iterations, so that a graph can be drawn to see the variation trend of accuracy:
    accuracys_train = []
    accuracys_test = []
    for i in range(int(num_iterations / step)):
        # Gradient descent, model parameters can be calculated iteratively:
        params, grads, costs = optimize(W, b, x_train, y_train, step, learning_rate, costs,i*step, print_cost)
        W = params['w']
        b = params['b']

        # Use the parameters learned to make predictions:
        prediction_train = predict(W, b, x_train)

        # Calculation accuracy, respectively in training set and test set:
        accuracy_train = 1 - np.mean(np.abs(prediction_train - y_train))
        print("Accuracy on train set:", accuracy_train)
        accuracys_train.append(accuracy_train)
    # To facilitate analysis and inspection, we store all the parameters and hyperparameters obtained into a dictionary and return them:
    d = {"costs": costs,
         "Y_prediction_train": prediction_train,
         "w": W,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations,
         "train_acy": accuracy_train,
         "accuracys_train": accuracys_train
         }
    # Use the parameters learned to make predictions:
    prediction_test = predict(W, b, x_test)

    # Calculation accuracy, respectively in training set and test set:
    accuracy_test = 1 - np.mean(np.abs(prediction_test - y_test))
    print("Accuracy on test set:", accuracy_test)

    return d


startTime = time.time()
X_train, Y_train, X_test, Y_test = load_data()
endLoadTime = time.time()
d = logistic_model(X_train, Y_train, X_test, Y_test, num_iterations=3000,
                   learning_rate=0.15, print_cost=True)
endTime = time.time()
print("load file time: %.2f\ncalculating time: %.2f \ntotal time: %.2f" % (
    endLoadTime - startTime, endTime - endLoadTime, endTime - startTime))

# first image: change of costs
plt.title("change of costs during iteration")
plt.ylabel('costs')
plt.xlabel('iterations*100')
plt.plot(d['costs'])
plt.show()

# draw the change of accuracy of training data during iteration
plt.ylabel("accuracy")
plt.title("change of accuracy of training data during iteration")
plt.xlabel('iterations*100')
plt.plot(d['accuracys_train'])
plt.show()
