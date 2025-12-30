import os
import torch
from torch.utils.data import Dataset
import glob
import config
import utils

# 오디오 데이터셋 클래스
class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        
        for class_idx, class_name in config.CLASSES.items():
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"경고: 클래스 디렉토리가 없습니다: {class_dir}")
                continue
                
            audio_files = []
            for ext in ['.wav', '.mp3', '.flac']:
                audio_files.extend(glob.glob(os.path.join(class_dir, f'*{ext}')))
            
            if not audio_files:
                print(f"경고: '{class_name}' 클래스에 오디오 파일이 없습니다.")
                continue
                
            print(f"'{class_name}' 클래스에서 {len(audio_files)}개 파일 발견")
            
            for audio_path in audio_files:
                try:
                    # 각 오디오 파일을 로드
                    audio = utils.load_audio(audio_path)
                    if audio is None or (isinstance(audio, torch.Tensor) and audio.numel() == 0):
                        print(f"경고: 오디오 파일을 로드할 수 없음: {audio_path}")
                        continue
                    
                    # 오디오 차원 확인
                    if isinstance(audio, torch.Tensor) and audio.dim() == 0:
                        print(f"경고: 오디오 텐서 차원이 0임: {audio_path}")
                        continue
                    
                    # 오디오가 길면 세그먼트로 분할
                    target_length = int(config.SAMPLE_RATE * config.WINDOW_SIZE)
                    if audio.size(0) > target_length:
                        # 오디오를 겹치는 세그먼트로 분할
                        segments = utils.audio_to_segments(audio)
                        for i, segment in enumerate(segments):
                            self.samples.append({
                                'path': audio_path,
                                'class': class_idx,
                                'segment': segment,
                                'is_segment': True,
                                'segment_idx': i
                            })
                    else:
                        # 오디오가 짧은 경우 그대로 사용
                        self.samples.append({
                            'path': audio_path,
                            'class': class_idx,
                            'is_segment': False
                        })
                except Exception as e:
                    print(f"오류: {audio_path} 파일 처리 중 예외 발생: {str(e)}")
                    continue
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['class']
        
        if sample.get('is_segment', False):
            # 미리 분할된 세그먼트 사용
            audio = sample['segment']
        else:
            # 짧은 오디오 파일 로드
            audio = utils.load_audio(sample['path'])
        
        # 잡음 및 바람소리 필터링 추가
        audio = utils.simple_noise_filter(audio)
        
        # 필요시 오디오 길이 조정 
        target_length = int(config.SAMPLE_RATE * config.WINDOW_SIZE)
        if audio.size(0) < target_length:
            padding = torch.zeros(target_length - audio.size(0))
            audio = torch.cat([audio, padding])
        elif audio.size(0) > target_length:
            # 세그먼트가 아직도 너무 길면 자르기
            audio = audio[:target_length]
        
        # 특징 추출
        features = utils.extract_features(audio)
        
        # 형태 변환: (n_mels, time) -> (1, n_mels, time) 
        features = torch.FloatTensor(features).unsqueeze(0)
        
        return features, torch.tensor(label, dtype=torch.long)