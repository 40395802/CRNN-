#  Real-time Accident Sound Detection System

CRNN(Convolutional Recurrent Neural Network) 기반 실시간 사고 음향 감지 시스템입니다. 마이크를 통해 실시간으로 오디오를 분석하고 사고 소리를 자동으로 감지합니다.

## 주요 기능

- **실시간 음향 감지**: 마이크 입력을 실시간으로 모니터링하여 사고 소리 감지
- **적응형 임계값**: 환경에 따라 자동으로 임계값을 조정하여 오탐지 최소화
- **시각화 대시보드**: Mel Spectrogram과 감지 확률을 실시간으로 시각화
- **자동 알림**: 사고 감지 시 자동 알림 및 오디오 저장
- **교차 검증**: K-Fold 교차 검증을 통한 모델 성능 평가

## 시스템 구조

```
CRNN Architecture:
Input Audio → CNN (Feature Extraction) → RNN (Temporal Pattern) → Classification
              ├─ 3 Conv Layers               ├─ GRU/LSTM             └─ Binary Output
              ├─ BatchNorm                   └─ Bidirectional           (Normal/Accident)
              └─ MaxPooling
```


## 요구사항

```
Python >= 3.8
torch >= 1.9.0
torchaudio >= 0.9.0
librosa >= 0.9.0
pyaudio
numpy
matplotlib
scikit-learn
pandas
scipy
```


## 프로젝트 구조

```
accident-detection/
├── config.py              # 설정 파일 (하이퍼파라미터, 경로 등)
├── model.py               # CRNN 모델 정의
├── dataset.py             # 데이터셋 클래스
├── utils.py               # 유틸리티 함수들
├── train.py               # 모델 학습 스크립트
├── realtime_detector.py   # 실시간 감지 시스템
├── cross_validation.py    # 교차 검증 스크립트
├── main.py                # 메인 실행 파일
├── data/                  # 학습 데이터
│   ├── 정상/              # 정상 음향 파일
│   └── 사고/              # 사고 음향 파일
└── saved/                 # 저장된 모델 및 결과
    ├── models/            # 학습된 모델
    └── results/           # 학습 결과 및 그래프
```


## 주요 설정 (config.py)

### 오디오 처리
- `SAMPLE_RATE`: 48000 Hz
- `WINDOW_SIZE`: 1.0초
- `OVERLAP`: 50%
- `N_MELS`: 64 (Mel frequency bins)

### 모델 파라미터
- `CNN_FILTERS`: [32, 64, 128]
- `RNN_HIDDEN_SIZE`: 128
- `RNN_TYPE`: 'GRU' (또는 'LSTM')
- `DROPOUT_RATE`: 0.3

### 학습 설정
- `BATCH_SIZE`: 128
- `EPOCHS`: 50
- `LEARNING_RATE`: 0.001
- `PATIENCE`: 5 (Early stopping)

### 감지 설정
- `PREDICTION_THRESHOLD`: 0.6 (기본 임계값)
- `CONSECUTIVE_FRAMES`: 2 (연속 감지 필요 프레임)
- `ALERT_COOLDOWN`: 10초 (재알림 쿨다운)

### 적응형 임계값
- `USE_ADAPTIVE_THRESHOLD`: False (기본값)
- `ADAPTIVE_FACTOR`: 1.0 (표준편차 가중치)
- `ADAPTIVE_MIN_THRESHOLD`: 0.5
- `ADAPTIVE_MAX_THRESHOLD`: 0.9

### 1. 적응형 임계값 시스템
환경 소음 수준에 따라 자동으로 임계값을 조정하여 오탐지를 줄입니다.

```python
# 평균 + (표준편차 × 가중치) 방식으로 임계값 계산
adaptive_threshold = mean(recent_probs) + factor × std(recent_probs)
```

### 2. 주파수 에너지 분석
특정 주파수 대역의 에너지를 분석하여 사고 소리 패턴을 사전 필터링합니다.

### 3. 노이즈 필터링
- RMS 정규화
- 저주파/고주파 필터링
- Butterworth 필터 적용

### 4. 논문
https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003261133


