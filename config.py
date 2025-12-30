import os
import torch

# 경로 
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
SAVED_DIR = os.path.join(ROOT_DIR, 'saved')
MODEL_DIR = os.path.join(SAVED_DIR, 'models')
RESULT_DIR = os.path.join(SAVED_DIR, 'results')
ACCIDENT_DIR = os.path.join(RESULT_DIR, 'accidents')

# 디렉토리
for directory in [DATA_DIR, SAVED_DIR, MODEL_DIR, RESULT_DIR, ACCIDENT_DIR]:
    os.makedirs(directory, exist_ok=True)

# 장치 
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')
# print(f"사용 장치: {DEVICE}")

# 오디오 처리 
SAMPLE_RATE = 48000     # sampling rate
WINDOW_SIZE = 1.0        # 1s window
OVERLAP = 0.5            # 50% overlap
N_FFT = 2048
HOP_LENGTH = 1024
N_MELS = 64
TARGET_RMS = 0.1        # RMS 정규화

# CRNN 모델 파라미터
CNN_FILTERS = [32, 64, 128]
CNN_KERNEL_SIZE = (3, 3)
CNN_POOL_SIZE = (2, 2)
RNN_HIDDEN_SIZE = 128
RNN_NUM_LAYERS = 2
RNN_TYPE = 'GRU'         
FC_SIZE = 64
DROPOUT_RATE = 0.3

# 학습 
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 5             # early stopping patience
VALIDATION_SPLIT = 0.2

# 추론 
PREDICTION_THRESHOLD = 0.6   # 기본 예측 임계값
CONSECUTIVE_FRAMES = 2       # 연속 프레임 감지 횟수
ALERT_COOLDOWN = 10          # 알림 쿨다운 시간
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'latest_model.pth')

# 적응형 임계값 설정
USE_ADAPTIVE_THRESHOLD = False    # 적응형 임계값 기본 활성화 여부
ADAPTIVE_WINDOW_SIZE = 30        # 적응형 임계값 계산에 사용할 프레임 수
ADAPTIVE_FACTOR = 1            # 표준편차 가중치
ADAPTIVE_MIN_THRESHOLD = 0.5     # 최소 임계값
ADAPTIVE_MAX_THRESHOLD = 0.9     # 최대 임계값

# 클래스 
CLASSES = {0: "정상", 1: "사고"}
NUM_CLASSES = len(CLASSES)

# 정규화 
WEIGHT_DECAY = 0.0005  # L2 정규화 
DROPOUT_RATE = 0.5     # 드롭아웃 비율