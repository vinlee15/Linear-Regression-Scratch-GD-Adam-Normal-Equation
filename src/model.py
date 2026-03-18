import numpy as np
import copy

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
    w_history = []
    cost_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    y_train = predict_price(x_train, w, b)
    temp_cost = compute_cost(w, y_train, y_real, lambda_)

    for t in range(num_iters):
        g_w, g_b = compute_gradient(w, x_train, y_train, y_real, lambda_)
        w = w - alpha * g_w
        b = b - alpha * g_b

        y_train = predict_price(x_train, w, b)
        cost = compute_cost(w, y_train, y_real, lambda_)

        if t % 100 == 0:
            print(f"Vòng {t:5}: Cost = {cost:.6f}")

        w_history.append(copy.deepcopy(w))
        cost_history.append(cost)

        diff = temp_cost - cost

        if 0 <= diff < epsilon:
            print(f"--- Thuật toán hội tụ tại vòng {t} Cost = {cost:.6f}")
            break

        if diff < 0:
            print(f"Cảnh báo: Alpha quá cao! Cost tăng từ {temp_cost:.6f} lên {cost:.6f} tại vòng {t}")
            alpha /= 3

        temp_cost = cost
    
    return w, b, w_history, cost_history

def adam_optimizer(w_in, b_in, x_train, y_real, alpha, num_iters, epsilon, lambda_):
    w_history = []
    cost_history = []
    eps = 1e-8
    beta1 = 0.9
    beta2 = 0.999
    w = copy.deepcopy(w_in)
    b = b_in
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_b = v_b = 0

    y_train = predict_price(x_train, w, b)
    temp_cost = compute_cost(w, y_train, y_real, lambda_)

    for t in range (num_iters):
        g_w, g_b = compute_gradient(w, x_train, y_train, y_real, lambda_)
        m_w = beta1 * m_w + (1 - beta1) * g_w
        m_b = beta1 * m_b + (1 - beta1) * g_b
        v_w = beta2 * v_w + (1 - beta2) * g_w ** 2
        v_b = beta2 * v_b + (1 - beta2) * g_b ** 2

        bias_correction1 = 1 - beta1**(t + 1) 
        bias_correction2 = 1 - beta2**(t+ 1) 
        m_w_h = m_w / bias_correction1
        m_b_h = m_b / bias_correction1
        v_w_h = v_w / bias_correction2
        v_b_h = v_b / bias_correction2

        w = w - alpha * m_w_h / (np.sqrt(v_w_h) + eps)
        b = b - alpha * m_b_h / (np.sqrt(v_b_h) + eps)

        y_train = predict_price(x_train, w, b)
        cost = compute_cost(w, y_train, y_real, lambda_)

        w_history.append(copy.deepcopy(w))
        cost_history.append(cost)

        if t % 100 == 0:
            print(f"Vòng {t:5}: Cost = {cost:.6f}")
    
        diff = temp_cost - cost

        if 0 <= diff < epsilon:
            print(f"--- Thuật toán hội tụ tại vòng {t} Cost = {cost:.6f} ---")
            break

        if diff < 0:
            print(f"Cảnh báo: Alpha quá cao! Cost tăng từ {temp_cost:.6f} lên {cost:.6f} tại vòng {t}")
            alpha /= 3

        temp_cost = cost 

    return w, b, w_history, cost_history

def normal_equation(x_train, y_real, lambda_):
    m, n = x_train.shape
    I = np.eye(n + 1)
    I[0, 0] = 0

    X = np.empty((m, n + 1))
    X[:, 0] = 1
    X[:, 1: ] = x_train 

    X_transpose = np.transpose(X)
    xtx_inv = np.linalg.pinv(np.dot(X_transpose, X) + lambda_ * I)
    xty = np.dot(X_transpose, y_real)
    w = np.dot(xtx_inv, xty)
    
    y_train = predict_price(x_train, w[1:, 0], w[0, 0])
    y_train = y_train.reshape(-1, 1)
    cost = compute_cost(w[1:, 0], y_train, y_real, lambda_)
    print(f"--- Thuật toán hội tụ tại Cost = {cost:.6f}")

    return w[1:, 0], w[0, 0]
