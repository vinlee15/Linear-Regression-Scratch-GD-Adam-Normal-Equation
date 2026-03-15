import numpy as np
import copy
import math

# w : (n, 1)
# b : scalar
# x : (m, n)
# y : (m, 1)

def predict_price(x, w, b):
    return np.dot(x, w) + b

def compute_cost(w, y_train, y_real, lambda_):
    m = y_real.shape[0]

    cost = y_train - y_real
    regularization = lambda_ * np.dot(np.transpose(w), w) / (2 * m)
    total_cost = np.dot(np.transpose(cost), cost) / (2 * m) + regularization

    return total_cost.item()

def compute_gradient(w, x_train, y_train, y_real, lambda_):
    m = y_real.shape[0]

    cost = (y_train - y_real).reshape(-1, 1)
    regularization = lambda_ * w / m 
    w_gradient = np.dot(np.transpose(x_train), cost) / m + regularization
    b_gradient = np.sum(cost) / m

    return w_gradient, b_gradient

def gradient_decent(w_in, b_in, x_train, y_real, alpha, num_iters, epsilon, lambda_):
    y_real.reshape(-1, 1)
    w = copy.deepcopy(w_in)
    b = b_in
    y_train = predict_price(x_train, w, b)
    temp_cost = compute_cost(w, y_train, y_real, lambda_)

    for i in range(num_iters):
        w_gradient, b_gradient = compute_gradient(w, x_train, y_train, y_real, lambda_)
        w = w - alpha * w_gradient
        b = b - alpha * b_gradient
        y_train = predict_price(x_train, w, b)

        cost = compute_cost(w, y_train, y_real, lambda_)

        if temp_cost - cost < epsilon:
            if cost > temp_cost:
                print("alpha quas lớn")
            break
        temp_cost = cost
    
    return w, b