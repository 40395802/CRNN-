import os
import sys
import argparse
import config
import utils
from train import train_accident_detector
from realtime_detector import start_realtime_detector
from cross_validation import k_fold_cross_validation
# 메인 함수
def main():
    parser = argparse.ArgumentParser(description='CRNN 기반 실시간 사고 감지 시스템')
    
    subparsers = parser.add_subparsers(dest='command', help='명령어')
    
    # 학습 명령어
    train_parser = subparsers.add_parser('train', help='모델 학습')
    train_parser.add_argument('--data_dir', type=str, default=config.DATA_DIR, help='데이터 디렉토리 경로')
    
    # 감지 명령어
    detect_parser = subparsers.add_parser('detect', help='실시간 감지')
    detect_parser.add_argument('--model', type=str, default=None, help='학습된 모델 파일 경로')
    detect_parser.add_argument('--threshold', type=float, default=config.PREDICTION_THRESHOLD, help='기본 사고 감지 임계값')
    detect_parser.add_argument('--consecutive', type=int, default=config.CONSECUTIVE_FRAMES, help='연속 감지 프레임 수')
    detect_parser.add_argument('--adaptive', action='store_true', help='적응형 임계값 사용 (기본: 비활성화)')
    detect_parser.add_argument('--fixed', action='store_true', help='고정 임계값 사용 (기본: 비활성화)')
    detect_parser.add_argument('--factor', type=float, default=2.0, help='적응형 임계값 계수 (표준편차 가중치)')
    
    # 초기화 명령어 
    init_parser = subparsers.add_parser('init', help='데이터 디렉토리 구조 초기화')
    
    cv_parser = subparsers.add_parser('cv', help='교차 검증')
    cv_parser.add_argument('--data_dir', type=str, default=config.DATA_DIR, help='데이터 디렉토리 경로')
    cv_parser.add_argument('--k', type=int, default=5, help='폴드 수')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("="*50)
        print("CRNN 모델 학습 시작")
        print("="*50)
        train_accident_detector(args.data_dir)
    
    elif args.command == 'detect':
        # 모델 파일 결정
        if args.model:
            model_path = args.model
        else:
            model_path = utils.find_latest_model()
            if not model_path:
                print("오류: 학습된 모델을 찾을 수 없습니다. 먼저 모델을 학습하세요.")
                print("사용법: python main.py train")
                return
        
        # 임계값 모드 결정
        use_adaptive = config.USE_ADAPTIVE_THRESHOLD
        if args.fixed:
            use_adaptive = False
        if args.adaptive:
            use_adaptive = True    
        
        print("="*50)
        print("실시간 사고 감지 시작")
        print("="*50)
        print(f"임계값 모드: {'적응형' if use_adaptive else '고정'}")
        
        from realtime_detector import RealtimeDetector
        detector = RealtimeDetector(
            model_path=model_path,
            threshold=args.threshold,
            consecutive_frames=args.consecutive
        )
        
        # 추가 설정
        detector.adaptive_mode = use_adaptive
        if hasattr(args, 'factor'):
            detector.adaptation_factor = args.factor
        
        detector.start()
    
    elif args.command == 'init':
        print("="*50)
        print("데이터 디렉토리 구조 초기화")
        print("="*50)
        utils.prepare_example_data()
        
    elif args.command == 'cv':
        print("="*50)
        print(f"{args.k}-Fold 교차 검증 시작")
        print("="*50)
        k_fold_cross_validation(args.data_dir, args.k)    
    
    else:
        if utils.check_model_exists():
            print("실시간 감지 모드로 시작합니다.")
            model_path = utils.find_latest_model()
            
            
            from realtime_detector import RealtimeDetector
            detector = RealtimeDetector(
                model_path=model_path,
                threshold=config.PREDICTION_THRESHOLD,
                consecutive_frames=config.CONSECUTIVE_FRAMES
            )
            
            # 기본값: 적응형 임계값 활성화
            detector.adaptive_mode = config.USE_ADAPTIVE_THRESHOLD
            detector.start()
            
        elif utils.check_data_exists():
            # 데이터가 존재하면 학습 모드로 시작
            print("학습 모드로 시작합니다.")
            train_accident_detector(config.DATA_DIR)
        else:
            print("사용법:")
            print("1. 데이터 구조 초기화: python main.py init")
            print("2. 모델 학습: python main.py train")
            print("3. 실시간 감지 (적응형 임계값): python main.py detect")
            print("4. 실시간 감지 (고정 임계값): python main.py detect --fixed")
            print("5. 임계값 계수 설정: python main.py detect --factor 1.5")
            
            parser.print_help()
            
            utils.prepare_example_data()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n프로그램 중단")
        sys.exit(0)
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)