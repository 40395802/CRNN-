import os
import time
from datetime import datetime
import numpy as np
import torch
import pyaudio
import queue
import threading
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
import torchaudio

import config
from model import CRNN
import utils

warnings.filterwarnings('ignore')

# 오디오 버퍼 클래스
class AudioBuffer:
    def __init__(self, buffer_size=int(config.SAMPLE_RATE * config.WINDOW_SIZE), overlap=config.OVERLAP):
        self.buffer_size = buffer_size
        self.overlap = overlap
        self.stride = int(buffer_size * (1 - overlap))
        
        # 초기 버퍼
        self.buffer = np.zeros(buffer_size, dtype=np.float32)
        self.is_full = False
    
    # 샘플 추가
    def add_samples(self, samples):
        # 샘플 수
        n_samples = len(samples)
        
        if n_samples >= self.buffer_size:
            # 새 샘플이 버퍼보다 크면 마지막 부분만 사용
            self.buffer = samples[-self.buffer_size:]
            self.is_full = True
        else:
            # 버퍼 시프트하고 새 샘플 추가
            shift = min(n_samples, self.stride)
            self.buffer = np.roll(self.buffer, -shift)
            self.buffer[-n_samples:] = samples
            
            # 최소 한 번은 버퍼가 가득 차야 함
            if not self.is_full:
                filled = np.sum(self.buffer != 0)
                self.is_full = filled >= self.buffer_size
   
    # 버퍼 가져오기
    def get_buffer(self):
        return self.buffer.copy() if self.is_full else None


# 실시간 감지 클래스
class RealtimeDetector:
    def __init__(self, model_path, threshold=config.PREDICTION_THRESHOLD, consecutive_frames=config.CONSECUTIVE_FRAMES):
        # 모델 로드
        self.device = config.DEVICE
        self.model = CRNN().to(self.device)
        self.model = utils.load_model(self.model, model_path)
        self.model.eval()
        
        # 설정
        self.base_threshold = threshold  # 기본 임계값 (초기값 또는 최소값으로 사용)
        self.current_threshold = threshold  # 현재 적용중인 적응형 임계값
        self.consecutive_frames = consecutive_frames
        self.alert_cooldown = config.ALERT_COOLDOWN
        
        # 적응형 임계값 관련 설정
        self.prob_window_size = config.ADAPTIVE_WINDOW_SIZE  # 최근 30개 프레임의 확률값 저장
        self.prob_window = deque(maxlen=self.prob_window_size)
        self.adaptation_factor = config.ADAPTIVE_FACTOR  # 표준편차의 가중치 (높을수록 더 엄격한 임계값)
        self.min_threshold = config.ADAPTIVE_MIN_THRESHOLD  # 최소 임계값
        self.max_threshold = config.ADAPTIVE_MAX_THRESHOLD  # 최대 임계값
        self.adaptive_mode = config.USE_ADAPTIVE_THRESHOLD  # 적응형 임계값 모드 활성화 여부
        
        # 오디오 설정
        self.sample_rate = config.SAMPLE_RATE
        self.window_size = config.WINDOW_SIZE
        self.overlap = config.OVERLAP
        
        # 오디오 버퍼
        self.audio_buffer = AudioBuffer(
            buffer_size=int(self.sample_rate * self.window_size),
            overlap=self.overlap
        )
        
        # 감지 상태
        self.detection_scores = deque(maxlen=self.consecutive_frames)
        self.detected = False
        self.last_alert_time = 0
        
        # 스트리밍 설정
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # 시각화 데이터
        self.mel_spec = None
        self.current_score = 0
        self.score_history = deque(maxlen=100)  # 100개 프레임 이력
        self.threshold_history = deque(maxlen=100)  # 임계값 이력 추가
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.mel_plot = None
        self.prob_line = None
        self.threshold_line = None
        self.status_text = None
        
    def analyze_frequency_energy(self, features):

        # 특정 주파수 대역 (가정: 사고 소리는 20~40 사이의 멜 빈에 집중)
        accident_freq_band = features[20:40, :]
        
        # 해당 대역의 평균 에너지 계산
        band_energy = np.mean(accident_freq_band)
        
        # 전체 스펙트로그램 에너지와 비교
        total_energy = np.mean(features)
        energy_ratio = band_energy / (total_energy + 1e-8)
        
        return energy_ratio > 0.5  # 에너지 비율 임계값 (조정 가능)    
    
    # 적응형 임계값 계산
    def calculate_adaptive_threshold(self):
        if len(self.prob_window) < 10:  # 충분한 데이터가 없으면 기본값 사용
            return self.base_threshold
        
        mean_prob = np.mean(self.prob_window)
        std_prob = np.std(self.prob_window)
        
        # 평균 + adaptation_factor * 표준편차
        adaptive_threshold = mean_prob + self.adaptation_factor * std_prob
        
        # 임계값 범위 제한
        adaptive_threshold = max(self.min_threshold, min(self.max_threshold, adaptive_threshold))
        
        return adaptive_threshold
    
    # PyAudio 콜백 함수
    def audio_callback(self, in_data, frame_count, time_info, status):
        # 바이트 데이터를 float32로 변환
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 리샘플링 필요한 경우 수행
        if hasattr(self, 'needs_resampling') and self.needs_resampling:
            # numpy 배열을 torch 텐서로 변환
            audio_tensor = torch.FloatTensor(audio_data)
            
            # 리샘플러 생성 
            if not hasattr(self, 'resampler'):
                self.resampler = torchaudio.transforms.Resample(
                    orig_freq=self.original_rate,
                    new_freq=config.SAMPLE_RATE
                )
            
            # 리샘플링 수행
            audio_tensor = self.resampler(audio_tensor)
            
            # 다시 numpy 배열로 변환
            audio_data = audio_tensor.numpy()
        
        # 오디오 큐에 데이터 추가
        self.audio_queue.put(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    # 오디오 처리 스레드
    def process_audio(self):
        while not self.stop_event.is_set():
            try:
                # 오디오 큐에서 데이터 가져오기
                try:
                    audio_data = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # RMS 정규화 적용
                audio_data = utils.apply_rms_normalization(audio_data)
                
                # 잡음 및 바람소리 필터링 
                audio_data = utils.simple_noise_filter(audio_data)
                
                # 오디오 버퍼에 데이터 추가
                self.audio_buffer.add_samples(audio_data)
                
                # 버퍼가 가득 차면 처리
                buffer_data = self.audio_buffer.get_buffer()
                if buffer_data is not None:
                    # 특징 추출
                    features = utils.extract_features(buffer_data)
                    self.mel_spec = features  # 시각화용 저장
                    
                    # 여기에 주파수 에너지 분석 추가
                    has_accident_energy = self.analyze_frequency_energy(features)
                    
                    # 에너지 패턴이 사고와 유사하지 않으면 처리 스킵
                    if not has_accident_energy:
                        self.current_score = 0.1  # 낮은 확률로 설정
                        self.score_history.append(self.current_score)
                        self.prob_window.append(self.current_score)
                        self.threshold_history.append(self.current_threshold)
                        self.detection_scores.append(False)
                        self.detected = False
                        continue
                    
                    # 이후 기존 모델 추론 코드 계속 실행...
                    features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(features_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        accident_prob = probabilities[0, 1].item()
                    
                    self.current_score = accident_prob
                    self.score_history.append(accident_prob)
                    
                    # 확률값 이력 업데이트
                    self.prob_window.append(accident_prob)
                    
                    # 적응형 임계값 계산 및 적용
                    if self.adaptive_mode:
                        self.current_threshold = self.calculate_adaptive_threshold()
                    else:
                        self.current_threshold = self.base_threshold
                    
                    # 임계값 이력 저장
                    self.threshold_history.append(self.current_threshold)
                    
                    # 사고 감지 로직 (현재 임계값 사용)
                    self.detection_scores.append(accident_prob >= self.current_threshold)
                    
                    # 연속된 프레임에서 임계값 이상으로 감지되면 사고로 판단
                    if sum(self.detection_scores) >= self.consecutive_frames:
                        current_time = time.time()
                        # 알림 쿨다운 확인
                        if not self.detected or (current_time - self.last_alert_time) > self.alert_cooldown:
                            self.detected = True
                            self.last_alert_time = current_time
                            
                            # 사고 알림
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            print(f"\n[{timestamp}] 사고 감지! 확률: {accident_prob:.4f}, 임계값: {self.current_threshold:.4f}")
                                
                            # 사고 기록 저장
                            accident_filename = f'accident_{timestamp.replace(":", "-").replace(" ", "_")}.npy'
                            np.save(
                                os.path.join(config.ACCIDENT_DIR, accident_filename),
                                buffer_data
                            )
                    else:
                        self.detected = False
            
            except Exception as e:
                print(f"오디오 처리 오류: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
            
    # 시각화 설정
    def setup_visualization(self):
        plt.ion()  
        
        # 전체 폰트 크기 설정 
        plt.rcParams.update({
            'font.size': 12,          # 기본 폰트 크기  
            'axes.titlesize': 14,     # 제목 폰트 크기 
            'axes.labelsize': 12,     # 축 라벨 폰트 크기
            'xtick.labelsize': 10,    # x축 눈금 폰트 크기  
            'ytick.labelsize': 10,    # y축 눈금 폰트 크기 
            'legend.fontsize': 10,    # 범례 폰트 크기 
            'figure.titlesize': 16    # 전체 제목 폰트 크기 
        })
        
        self.fig = plt.figure(figsize=(12, 8))
        
        # 멜 스펙트로그램 서브플롯
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.mel_plot = self.ax1.imshow(
            np.zeros((config.N_MELS, 100)),
            aspect='auto',
            origin='lower',
            cmap='viridis'
        )
        self.ax1.set_title('Mel Spectrogram', fontsize=28) # default 16
        self.ax1.set_ylabel('Frequency', fontsize=26)   # default 14
        self.ax1.set_xlabel('Time Frame', fontsize=26) # default 14
        
        # 확률 이력 서브플롯
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        self.prob_line, = self.ax2.plot([], [], 'b-', linewidth=2, label='Model Inference Output')
        self.threshold_line, = self.ax2.plot([], [], 'r--', linewidth=2, label='Threshold')
        self.ax2.set_title('Accident Detection', fontsize=28) # default 16
        self.ax2.set_ylabel('Output', fontsize=26)  # default 14
        self.ax2.set_xlabel('Time Frame', fontsize=26) # default 14
        self.ax2.set_ylim(0, 1)
        self.ax2.set_xlim(0, 100)
        self.ax2.grid(True)
        self.ax2.legend(loc='upper right', fontsize=20) # default 12
        
        # 상태 텍스트 
        self.status_text = self.ax2.text(0.02, 0.95, '', transform=self.ax2.transAxes, fontsize=16, 
                            bbox=dict(facecolor='white', alpha=0.8)) # default 14
        
        plt.tight_layout()
        self.fig.canvas.draw_idle()
        
    # 시각화 업데이트
    def update_visualization(self):
        # 멜 스펙트로그램 업데이트
        if self.mel_spec is not None:
            self.mel_plot.set_data(self.mel_spec)
            self.mel_plot.set_clim(vmin=np.min(self.mel_spec), vmax=np.max(self.mel_spec))
        
        # 확률 이력 및 임계값 이력 업데이트
        score_list = list(self.score_history)
        threshold_list = list(self.threshold_history)
        
        if score_list:
            x = np.arange(len(score_list))
            self.prob_line.set_data(x, score_list)
            
            if threshold_list:
                if len(threshold_list) < len(score_list):
                    # 확률값 이력과 임계값 이력의 길이가 다를 수 있음
                    padding = [threshold_list[0]] * (len(score_list) - len(threshold_list))
                    padded_thresholds = padding + threshold_list
                    self.threshold_line.set_data(x, padded_thresholds)
                else:
                    self.threshold_line.set_data(x, threshold_list)
            
            self.ax2.set_xlim(0, max(99, len(score_list)-1))
        
        # 상태 텍스트 업데이트
        status = "Status: " + ("ACCIDENT DETECTED!" if self.detected else "Normal")
        prob_text = f"Current Output: {self.current_score:.4f}"
        threshold_text = f"Current threshold: {self.current_threshold:.4f}"
        adaptive_text = "Mode: Adaptive" if self.adaptive_mode else "Mode: Fixed"
        
        self.status_text.set_text(f"{status}\n{prob_text}\n{threshold_text}\n{adaptive_text}")
        self.status_text.set_color('red' if self.detected else 'green')
        
        # 그래프 갱신
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    # 시작
    def start(self):
        try:
            # PyAudio 초기화
            p = pyaudio.PyAudio()
            
            # 사용 가능한 디바이스 확인
            info = p.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')
            input_devices = []
            
            for i in range(num_devices):
                device_info = p.get_device_info_by_host_api_device_index(0, i)
                if device_info.get('maxInputChannels') > 0:
                    input_devices.append((i, device_info))
            
            if not input_devices:
                print("오류: 입력 장치를 찾을 수 없습니다.")
                p.terminate()
                return
            
            # 기본 입력 장치
            input_device_index = p.get_default_input_device_info()['index']
            device_info = p.get_device_info_by_index(input_device_index)
            
            # 마이크의 원래 샘플레이트 확인
            original_rate = int(device_info.get('defaultSampleRate', config.SAMPLE_RATE))
            print(f"설정된 샘플레이트: {config.SAMPLE_RATE}Hz")
            
            # 리샘플링 필요 여부 확인
            self.needs_resampling = original_rate != config.SAMPLE_RATE
            self.original_rate = original_rate
            
            # 스트림 열기 (마이크의 원래 샘플레이트 사용)
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=original_rate,  # 마이크의 원래 샘플레이트 사용
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=1024,
                stream_callback=self.audio_callback
            )
            
            print(f"오디오 입력 장치: {p.get_device_info_by_index(input_device_index)['name']}")
            print(f"임계값 모드: {'적응형' if self.adaptive_mode else '고정'}")
            print(f"기본 임계값: {self.base_threshold:.2f}")
            print(f"적응 계수 (표준편차 가중치): {self.adaptation_factor:.2f}")
            print("마이크 스트리밍 시작...")
            
            # 시각화 설정
            self.setup_visualization()
            
            # 처리 스레드 시작
            processor_thread = threading.Thread(target=self.process_audio)
            processor_thread.daemon = True
            processor_thread.start()
            
            # 시각화 루프
            try:
                while not self.stop_event.is_set():
                    self.update_visualization()
                    plt.pause(0.1)
            except KeyboardInterrupt:
                print("\n감지 중단...")
                self.stop_event.set()
            
            # 스트림 종료
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # 스레드 종료 신호
            self.stop_event.set()
            processor_thread.join(timeout=1.0)
                    
            plt.close('all')            
            plt.ioff()
                
        except Exception as e:
            print(f"실시간 감지 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            
# 실시간 감지 시작
def start_realtime_detector(model_path, threshold=config.PREDICTION_THRESHOLD, consecutive_frames=config.CONSECUTIVE_FRAMES, adaptive_mode=True):
    print("실시간 사고 감지 시스템 시작...")
    print(f"모델 파일: {model_path}")
    print(f"기본 임계값: {threshold}")
    print(f"임계값 모드: {'적응형' if adaptive_mode else '고정'}")
    print(f"연속 프레임: {consecutive_frames}")
    print("Ctrl+C를 누르면 종료됩니다.")
    
    detector = RealtimeDetector(
        model_path=model_path,
        threshold=threshold,
        consecutive_frames=consecutive_frames
    )
    
    # 적응형 임계값 모드 설정
    detector.adaptive_mode = adaptive_mode
    
    detector.start()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='실시간 사고 감지 시스템')
    parser.add_argument('--model', type=str, default=None, help='학습된 모델 파일 경로')
    parser.add_argument('--threshold', type=float, default=config.PREDICTION_THRESHOLD, help='기본 사고 감지 임계값')
    parser.add_argument('--consecutive', type=int, default=config.CONSECUTIVE_FRAMES, help='연속 감지 프레임 수')
    parser.add_argument('--adaptive', type=bool, default=True, help='적응형 임계값 사용 여부')
    parser.add_argument('--adaptation_factor', type=float, default=2.0, help='적응형 임계값 계수 (표준편차 가중치)')
    
    args = parser.parse_args()
    
    # 모델 파일 결정
    if args.model:
        model_path = args.model
    else:
        model_path = utils.find_latest_model()
        if not model_path:
            print("오류: 학습된 모델을 찾을 수 없습니다. 먼저 모델을 학습하세요.")
            exit(1)
    
    detector = RealtimeDetector(
        model_path=model_path,
        threshold=args.threshold,
        consecutive_frames=args.consecutive
    )
    
    # 추가 설정
    detector.adaptive_mode = args.adaptive
    detector.adaptation_factor = args.adaptation_factor
    
    detector.start()