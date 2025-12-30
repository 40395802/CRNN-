import torch
import torch.nn as nn
import torch.nn.functional as F
import config

# CRNN 모델 정의
class CRNN(nn.Module):
    # 초기화
    def __init__(self):
        super(CRNN, self).__init__()
        
        # CNN 레이어
        self.conv1 = nn.Conv2d(1, config.CNN_FILTERS[0], kernel_size=config.CNN_KERNEL_SIZE, padding=1)
        self.bn1 = nn.BatchNorm2d(config.CNN_FILTERS[0])
        self.pool1 = nn.MaxPool2d(config.CNN_POOL_SIZE)
        
        self.conv2 = nn.Conv2d(config.CNN_FILTERS[0], config.CNN_FILTERS[1], kernel_size=config.CNN_KERNEL_SIZE, padding=1)
        self.bn2 = nn.BatchNorm2d(config.CNN_FILTERS[1])
        self.pool2 = nn.MaxPool2d(config.CNN_POOL_SIZE)
        
        self.conv3 = nn.Conv2d(config.CNN_FILTERS[1], config.CNN_FILTERS[2], kernel_size=config.CNN_KERNEL_SIZE, padding=1)
        self.bn3 = nn.BatchNorm2d(config.CNN_FILTERS[2])
        self.pool3 = nn.MaxPool2d(config.CNN_POOL_SIZE)
        
        self.dropout1 = nn.Dropout(config.DROPOUT_RATE)
        self.dropout2 = nn.Dropout(config.DROPOUT_RATE)
        self.dropout3 = nn.Dropout(config.DROPOUT_RATE)        
        
        # 시퀀스 길이 및 특징 차원 계산 
        self._calculate_output_dim()
        
        # RNN 레이어 GRU 
        self.rnn_type = config.RNN_TYPE
        if config.RNN_TYPE == 'GRU':
            self.rnn = nn.GRU(
                input_size=640,  # 실제 CNN 출력 크기
                hidden_size=config.RNN_HIDDEN_SIZE,
                num_layers=config.RNN_NUM_LAYERS,
                batch_first=True,
                bidirectional=True,
                dropout=config.DROPOUT_RATE if config.RNN_NUM_LAYERS > 1 else 0
            )
        else:  # LSTM
            self.rnn = nn.LSTM(
                input_size=640,  # 실제 CNN 출력 크기
                hidden_size=config.RNN_HIDDEN_SIZE,
                num_layers=config.RNN_NUM_LAYERS,
                batch_first=True,
                bidirectional=True,
                dropout=config.DROPOUT_RATE if config.RNN_NUM_LAYERS > 1 else 0
            )
        
        # 완전연결 레이어
        self.fc1 = nn.Linear(config.RNN_HIDDEN_SIZE * 2, config.FC_SIZE)  # *2는 양방향 RNN 때문
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        self.fc2 = nn.Linear(config.FC_SIZE, config.NUM_CLASSES)
        
    # CNN 출력 형태 계산
    def _calculate_output_dim(self):
        # 원래 입력 크기 (멜 빈, 시간 프레임)
        freq_bins = config.N_MELS  # 64
        # 실제 시간 프레임 수 (오디오 길이에 따라 결정됨)
        time_frames = int(config.SAMPLE_RATE * config.WINDOW_SIZE / config.HOP_LENGTH) + 1  # 32
        
        # CNN 통과 후 크기 계산 (3번의 2x2 풀링)
        time_frames = time_frames // 8  # 3번의 2x2 풀링 후 (32 -> 4)
        freq_bins = freq_bins // 8      # 3번의 2x2 풀링 후 (64 -> 8)
        
        self.cnn_output_length = time_frames
        self.cnn_feature_size = freq_bins * config.CNN_FILTERS[2]
        
        
    def forward(self, x):
        # CNN 레이어 통과
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x) 
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # CNN 출력 형태 변환
        batch_size, channels, time, freq = x.size()
        # print(f"CNN 출력 shape: {x.shape}")  # 디버깅 출력   
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, time, channels * freq)
        # print(f"RNN 입력 shape: {x.shape}")  # 디버깅 출력
        
        # RNN 통과
        if self.rnn_type == 'GRU':
            x, _ = self.rnn(x)
        else:
            x, (_, _) = self.rnn(x)
        
        # 마지막 시퀀스 출력 사용
        x = x[:, -1, :]
        
        # 완전연결 레이어 통과
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x 