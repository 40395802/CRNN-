import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm):
    """
    혼돈 행렬만 그리는 최소한의 코드
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=16)
    plt.colorbar()
    
    classes = ['0', '1']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=16)
    plt.yticks(tick_marks, classes, fontsize=16)
    
    # 행렬 내부에 숫자 표시
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=16)
    
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.show()

# 혼돈 행렬 데이터
confusion_matrix = np.array([
    [17610,27],
    [104, 4610]
])

# 혼돈 행렬 그리기
plot_confusion_matrix(confusion_matrix)