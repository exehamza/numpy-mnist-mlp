import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("./train.csv")
data = np.array(data)

m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0].astype(int)
X_dev = data_dev[1:n] / 255.0

data_train = data[1000:m].T
Y_train = data_train[0].astype(int)
X_train = data_train[1:n] / 255.0

def init_params():
    W1 = np.random.randn(64, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((64, 1))
    W2 = np.random.randn(10, 64) * np.sqrt(2 / 64)
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return (Z > 0).astype(int)

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((10, Y.size))
    for i in range(Y.size):
        one_hot_Y[Y[i], i] = 1
    return one_hot_Y

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = (1/m) * dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 50 == 0:
            acc = get_accuracy(get_predictions(A2), Y)
            print(f"Iteration {i} | Accuracy: {acc:.4f}")

    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.01, iterations=500)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)

def test_prediction(index):
    current_image = X_train[:, index].reshape(784, 1)
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = Y_train[index]

    print("Prediction:", prediction[0])
    print("Label:", label)

    plt.imshow(current_image.reshape(28, 28), cmap="gray")
    plt.show()


test_prediction(0)
test_prediction(1)
test_prediction(2)
test_prediction(3)
test_prediction(4)
test_prediction(5)
test_prediction(6)
test_prediction(7)
test_prediction(8)
test_prediction(9)
test_prediction(10)
test_prediction(11)
test_prediction(12)
test_prediction(13)
test_prediction(14)
test_prediction(15)