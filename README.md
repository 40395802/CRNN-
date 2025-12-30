
pip install -r requirements.txt

project3.2/                      # 프로젝트 루트 디렉토리
│
├── config.py                    # 설정 변수 및 하이퍼파라미터 정의
├── dataset.py                   # 데이터셋 클래스 및 데이터 로딩 처리
├── realtime_detector.py         # 실시간 오디오 감지 구현
├── utils.py                     # 유틸리티 함수 
├── cross_validation.py          # K-fold 교차 검증 구현
├── model.py                     # CRNN 모델 구조 정의
├── train.py                     # 모델 학습 및 검증 루틴
├── main.py                      # 프로그램 동작
├── requirements.txt             # 필요한 Python 패키지 목록
├── README.md                    # 프로젝트 문서 및 사용 설명서
│
├── data/                        # 데이터 디렉토리
│   ├── 정상/                    # 정상 소리 오디오 샘플 
│   │           
│   │  
│   │   
│   │
│   └── 사고/                    # 사고 소리 오디오 샘플 
│             
│      
│      
│
└── saved/                       # 저장된 결과물
    ├── models/                  # 학습된 모델 저장 위치
    │   ├── latest_model.pth     # 가장 최근 학습된 모델
    │   └── crnn_model_20250301_120000.pth  # 타임스탬프가 있는 모델 파일
    │   
    │
    └── results/                 # 학습 및 평가 결과 저장 위치
        ├── learning_curves_20250301_120000.png  # 학습 곡선 그래프
        ├── confusion_matrix_20250301_120000.png  # 혼동 행렬 시각화
        ├── metrics_20250301_120000.csv  # 성능 지표 데이터
        └── accidents/           # 감지된 사고 오디오 데이터 저장
                └── accident_2025-03-01_12-30-45.npy  # 감지된 사고 데이터
            