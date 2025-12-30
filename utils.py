import os
import numpy as np
import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import glob
import config

# 오디오 로드
def load_audio(file_path, sr=config.SAMPLE_RATE):
    try:
        audio, sr_orig = torchaudio.load(file_path)
        
        # 비어있는 파일 확인
        if audio.numel() == 0:
            print(f"경고: 비어있는 오디오 파일: {file_path}")
            return torch.zeros(int(config.SAMPLE_RATE * config.WINDOW_SIZE))
        # 샘플링 48000
        if sr_orig != sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr_orig, new_freq=sr)
            audio = resampler(audio)
        
        # 모노로 변환 
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # 텐서 차원 확인 및 보정
        if audio.dim() == 1:
            # RMS 정규화 적용
            audio = apply_rms_normalization(audio)
            return audio
        elif audio.dim() > 1:
            audio = audio.squeeze(0)
            # RMS 정규화 적용
            audio = apply_rms_normalization(audio)
            return audio  
        else:
            print(f"경고: 유효하지 않은 차원의 오디오 데이터: {file_path}")
            return torch.zeros(int(config.SAMPLE_RATE * config.WINDOW_SIZE))
            
    except Exception as e:
        print(f"오디오 로드 오류 ({file_path}): {str(e)}")
        return torch.zeros(int(config.SAMPLE_RATE * config.WINDOW_SIZE))
    
def simple_noise_filter(audio, sr=config.SAMPLE_RATE):
    import numpy as np
    import librosa
    from scipy.signal import butter, filtfilt
    
    # 텐서를 numpy로 변환
    is_tensor = isinstance(audio, torch.Tensor)
    if is_tensor:
        audio = audio.numpy()
    
    # 고주파 필터링 (바람소리 제거용)
    cutoff = 2000  # 2kHz 이상 감쇠
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    
    # 버터워스 로우패스 필터 설계
    order = 4
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # 필터 적용 (zero-phase 필터링으로 위상 왜곡 방지)
    filtered_audio = filtfilt(b, a, audio)
    
    # 정규화
    original_rms = np.sqrt(np.mean(audio**2))
    filtered_rms = np.sqrt(np.mean(filtered_audio**2))
    
    if filtered_rms > 1e-6:  # 0으로 나누기 방지
        filtered_audio = filtered_audio * (original_rms / filtered_rms)
    
    # 원래 타입으로 변환 (수정된 부분)
    if is_tensor:
        filtered_audio = torch.from_numpy(filtered_audio.copy()).float()
    
    return filtered_audio 
    
# RMS 정규화 함수
def apply_rms_normalization(audio, target_rms=0.1):
    if isinstance(audio, torch.Tensor):
        # 현재 RMS 값 계산
        current_rms = torch.sqrt(torch.mean(audio ** 2))
        
        # RMS가 0이면 정규화 불필요
        if current_rms < 1e-8:
            return audio
            
        # 스케일 팩터 계산
        scale_factor = target_rms / current_rms
        
        # 오디오 신호 정규화
        normalized_audio = audio * scale_factor
        
        return normalized_audio
    else:
        # numpy 배열인 경우
        current_rms = np.sqrt(np.mean(np.square(audio)))
        
        if current_rms < 1e-8:
            return audio
            
        scale_factor = target_rms / current_rms
        normalized_audio = audio * scale_factor
        
        return normalized_audio
    
    
# 특징 추출
def extract_features(audio, sr=config.SAMPLE_RATE, n_fft=config.N_FFT, 
                     hop_length=config.HOP_LENGTH, n_mels=config.N_MELS):
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
    
    # 멜 스펙트로그램 추출
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels
    )
    
    # 로그 스케일 변환
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    
    # 정규화 
    normalized = (log_mel_spectrogram - log_mel_spectrogram.mean()) / (log_mel_spectrogram.std() + 1e-8)
    
    return normalized

# 오디오를 세그먼트로 분할
def audio_to_segments(audio, sr=config.SAMPLE_RATE, window_size=config.WINDOW_SIZE, overlap=config.OVERLAP):
    samples_per_window = int(window_size * sr)
    stride = int(samples_per_window * (1 - overlap))
    
    # 오디오 길이 확인
    if isinstance(audio, torch.Tensor):
        audio_length = audio.shape[0]
    else:
        audio_length = len(audio)
    
    # 세그먼트 생성
    segments = []
    for start in range(0, audio_length - samples_per_window + 1, stride):
        if isinstance(audio, torch.Tensor):
            segment = audio[start:start + samples_per_window]
        else:
            segment = audio[start:start + samples_per_window]
        segments.append(segment)
    
    return segments

# 모델 저장
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"모델 저장 완료: {path}")
    
    # 최신 모델 링크 업데이트
    latest_model_path = os.path.join(os.path.dirname(path), 'latest_model.pth')
    if os.path.exists(latest_model_path):
        os.remove(latest_model_path)
    torch.save(model.state_dict(), latest_model_path)

# 모델 로드
def load_model(model, path):
    try:
        state_dict = torch.load(path, map_location=config.DEVICE)
        model.load_state_dict(state_dict)
        print(f"모델 로드 완료: {path}")
        return model
    except Exception as e:
        print(f"모델 로드 오류: {str(e)}")
        return model

# 성능 지표 계산
def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

# 학습 곡선 그리기
def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    plt.figure(figsize=(12, 5))
    
    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy', marker='o')
    plt.plot(val_accs, label='Validation Accuracy', marker='s')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"학습 곡선 저장 완료: {save_path}")
    
    plt.close()

# 혼동 행렬 그리기
def plot_confusion_matrix(cm, classes=list(config.CLASSES.values()), save_path=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"혼동 행렬 저장 완료: {save_path}")
    
    plt.close()

# 성능 지표 저장
def save_metrics(metrics, save_path):
    # 혼동 행렬을 제외한 지표들만 CSV로 저장
    metrics_dict = {k: [v] for k, v in metrics.items() if k != 'confusion_matrix'}
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(save_path, index=False)
    print(f"성능 지표 저장 완료: {save_path}")

# 최신 모델 찾기
def find_latest_model():
    if os.path.exists(config.DEFAULT_MODEL_PATH):
        return config.DEFAULT_MODEL_PATH
    
    # 모델 디렉토리에서 모든 .pth 파일 검색
    model_files = glob.glob(os.path.join(config.MODEL_DIR, '*.pth'))
    if not model_files:
        return None
    
    # 파일 수정 시간 기준으로 정렬
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

# 모델 존재 여부 확인
def check_model_exists():
    return find_latest_model() is not None

# 데이터 존재 여부 확인
def check_data_exists():
    for _, class_name in config.CLASSES.items():
        class_dir = os.path.join(config.DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            return False
        
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac']:
            audio_files.extend(glob.glob(os.path.join(class_dir, f'*{ext}')))
        
        if not audio_files:
            return False
    
    return True

# 예제 데이터 생성
def prepare_example_data():
    for class_idx, class_name in config.CLASSES.items():
        class_dir = os.path.join(config.DATA_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    print(f"데이터 디렉토리 구조가 {config.DATA_DIR}에 생성되었습니다.")
    print("다음 구조로 오디오 파일(.wav, .mp3, .flac)을 배치하세요:")
    for class_idx, class_name in config.CLASSES.items():
        print(f"  - {os.path.join(config.DATA_DIR, class_name)}/")
        print(f"      - audio1.wav")
        print(f"      - audio2.wav")
        print(f"      - ...")
        
# 정밀도-재현율      
def plot_precision_recall_curve(y_true, y_prob, save_path=None):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.fill_between(recall, precision, alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AP={ap:.4f})')
    plt.grid(True)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    
    if save_path:
        plt.savefig(save_path)
        print(f"정밀도-재현율 곡선 저장 완료: {save_path}")
    
    plt.close()
# F1 
def plot_f1_score(y_true, y_prob, save_path=None):
    from sklearn.metrics import f1_score
    
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    # 최적 임계값 찾기
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    best_f1 = f1_scores[best_threshold_idx]
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_scores, 'r-', linewidth=2)
    plt.axvline(x=best_threshold, color='k', linestyle='--')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score vs Threshold (Best: {best_f1:.4f} at {best_threshold:.2f})')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"F1 점수 그래프 저장 완료: {save_path}")
    
    plt.close()
# ROC-AUC
def plot_roc_curve(y_true, y_prob, save_path=None):
    from sklearn.metrics import roc_curve, roc_auc_score
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')  # 대각선
    plt.fill_between(fpr, tpr, alpha=0.2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC={auc:.4f})')
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    if save_path:
        plt.savefig(save_path)
        print(f"ROC 곡선 저장 완료: {save_path}")
    
    plt.close()    