import numpy as np
import pandas as pd 
from src import preprocess_data,  gradient_decent, predict_price, save_learning_curve

def main():
    epsilon = 1e-7
    x_train, y_real, x_test, test_id = preprocess_data('data/train.csv', 'data/test.csv', epsilon)

    b = 0
    w = np.zeros((x_train.shape[1], 1))
    alpha = 0.06
    num_iters = 10000
    lambda_ = 0.1

    w_final, b_final, w_history, cost_history = gradient_decent(w, b, x_train, y_real, alpha, num_iters, epsilon, lambda_)
    save_learning_curve(cost_history, w_history, title="Gradient Descent")

    y_log = predict_price(x_test, w_final, b_final)
    y_final = np.expm1(y_log.astype(float))

    submission = pd.DataFrame({
        'Id': test_id,
        'SalePrice': y_final.flatten()
    })

    submission.to_csv('data/submission_gd.csv', index=False)
    
if __name__ == "__main__":
    main()