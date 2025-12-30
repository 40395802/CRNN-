import numpy as np
import matplotlib.pyplot as plt

# 샘플 데이터 (그래프의 값과 비슷하게 설정)
epochs = np.arange(0, 23)  # 0부터 22까지의 에폭
train_losses = [0.38, 0.16, 0.12, 0.09, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04, 0.03, 0.04, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.03]
val_losses = [1.8, 1.15, 0.78, 0.85, 0.70, 0.45, 0.33, 0.33, 0.22, 0.40, 0.28, 0.27, 0.28, 0.07, 0.09, 0.20, 0.06, 0.12, 0.09, 0.12, 0.08, 0.03, 0.04]

# 학습 곡선만 그리기
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, 'o-', color='#1f77b4', label='Train Loss')  # 파란색
plt.plot(epochs, val_losses, 's-', color='#ff7f0e', label='Validation Loss')  # 주황색

plt.title('Loss Curves', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# 파일로 저장
plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')

plt.show()