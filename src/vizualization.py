import matplotlib.pyplot as plt
import numpy as np

def save_learning_curve(cost_history, w_history, title="Learning Curve", filename="learning_curve.png"):
    w_norm_history = [np.linalg.norm(w) for w in w_history]
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(cost_history, color='#2c3e50', linewidth=2)
    ax1.set_title(f'{title}: Cost Function', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Vòng lặp')
    ax1.set_ylabel('Cost')
    ax1.set_yscale('log') 

    ax2.plot(w_norm_history, color='#e67e22', linewidth=2)
    ax2.set_title(f'{title}: Weight Norm (||w||)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Vòng lặp')
    ax2.set_ylabel('Độ lớn trọng số')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✅ Biểu đồ đã được lưu tại: {filename}")
    plt.close()