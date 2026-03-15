import numpy as np
import pandas as pd 
from src import preprocess_data,  gradient_decent, predict_price, adam_optimizer, save_learning_curve

def main():
    epsilon = 1e-7
    b = 0
    num_iters = 10000
    lambda_ = 0.1

    x_train, y_real, x_test, test_id = preprocess_data('data/train.csv', 'data/test.csv', epsilon)
    w = np.zeros((x_train.shape[1], 1))

    print("\nCHỌN THUẬT TOÁN TỐI ƯU:")
    print("0) Gradient Descent (Truyền thống)")
    print("1) Adam Optimizer (Nâng cao)")
    
    choice = input("Nhập lựa chọn của bạn (0/1): ")

    if choice == "0":
        alpha=0.06
        print(f"Đang chạy Gradient Descent với alpha={alpha}...")
        w_final, b_final, w_history, cost_history = gradient_decent(w, b, x_train, y_real, alpha, num_iters, epsilon, lambda_)
        save_learning_curve(cost_history, w_history, title="Gradient Descent")
    elif choice == "1":
        alpha=0.003
        print(f"Đang chạy Adam Optimizer với alpha={alpha}...")
        w_final, b_final, w_history, cost_history = adam_optimizer(w, b, x_train, y_real, alpha, num_iters, epsilon, lambda_)
        save_learning_curve(cost_history, w_history, title="Adam Optimizer")
    else:
        print("Lựa chọn không hợp lệ. Thoát chương trình.")
        return

    y_log = predict_price(x_test, w_final, b_final)
    y_final = np.expm1(y_log.astype(float))

    submission = pd.DataFrame({
        'Id': test_id,
        'SalePrice': y_final.flatten()
    })

    submission.to_csv('data/submission.csv', index=False)
    
if __name__ == "__main__":
    main()