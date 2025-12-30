import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import f1_score, confusion_matrix 
import numba
import config
from model import CRNN
from dataset import AudioDataset
import utils
# Focal Loss 구현
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')
        
        # 오류율 추적을 위한 변수 추가
        self.fp_rate = None
        self.fn_rate = None
        
    def forward(self, inputs, targets):
        # 기존 Cross Entropy Loss 계산
        ce_loss = self.ce(inputs, targets)
        
        # 소프트맥스 확률 계산
        probs = torch.softmax(inputs, dim=1)
        
        # 정답 클래스의 예측 확률 추출
        batch_size = inputs.size(0)
        pt = probs[torch.arange(batch_size), targets]
        
        # Focal 가중치 계산 (1-pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # 현재 오류율에 따라 알파 조정
        current_alpha = self.alpha
        if self.fp_rate is not None and self.fn_rate is not None and self.fn_rate > 0:
            # 거짓양성이 거짓음성보다 많을 경우 알파 증가 (사고 클래스 가중치 증가)
            error_ratio = self.fp_rate / self.fn_rate
            if error_ratio > 1:
                # 로그 스케일로 조정치 계산 (오류 비율이 10배면 0.05 증가, 100배면 0.1 증가)
                adjustment = min(0.1, np.log10(error_ratio) * 0.05)
                current_alpha = min(0.95, self.alpha + adjustment)
            elif error_ratio < 1:
                # 거짓음성이 더 많은 경우 (드문 경우지만 처리)
                adjustment = max(-0.1, -np.log10(1/error_ratio) * 0.05)
                current_alpha = max(0.7, self.alpha + adjustment)
        
        # 클래스 불균형 가중치 (alpha)
        alpha_weight = torch.ones_like(targets).float().to(inputs.device)
        alpha_weight[targets == 1] = current_alpha  # 사고 클래스
        alpha_weight[targets == 0] = 1 - current_alpha  # 정상 클래스
        
        # 가중치 적용된 손실 계산
        loss = alpha_weight * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
    def update_error_rates(self, val_labels, val_preds):
        from sklearn.metrics import confusion_matrix
        
        try:
            cm = confusion_matrix(val_labels, val_preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            # 오류율 계산
            total_negative = tn + fp
            total_positive = tp + fn
            
            self.fp_rate = fp / max(1, total_negative)
            self.fn_rate = fn / max(1, total_positive)
            
            # 오류율 출력
            adjusted_alpha = self.alpha
            if self.fn_rate > 0:
                error_ratio = self.fp_rate / self.fn_rate
                if error_ratio > 1:
                    adjustment = min(0.1, np.log10(error_ratio) * 0.05)
                    adjusted_alpha = min(0.95, self.alpha + adjustment)
                elif error_ratio < 1:
                    adjustment = max(-0.1, -np.log10(1/error_ratio) * 0.05)
                    adjusted_alpha = max(0.7, self.alpha + adjustment)
                    
            print(f"오류율 업데이트: FP={self.fp_rate:.4f}, FN={self.fn_rate:.4f}, 기본 Alpha={self.alpha:.4f}, 조정 Alpha={adjusted_alpha:.4f}")
        except Exception as e:
            print(f"오류율 계산 중 오류 발생: {str(e)}")

# 학습 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=config.EPOCHS, patience=config.PATIENCE):
    # 결과 저장용 리스트
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 조기 종료 설정
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 이전 에폭의 검증 결과로 오류율 업데이트 (첫 에폭 제외)
        if epoch > 0 and hasattr(criterion, 'update_error_rates'):
            # 검증 데이터에서 예측 수행
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    _, predicted = torch.max(outputs, 1)
                    
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            # 오류율 업데이트
            criterion.update_error_rates(val_labels, val_preds)
            
        # 학습 모드
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 학습 루프
        pbar = tqdm(train_loader, desc=f'에폭 {epoch+1}/{num_epochs} [학습]')
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            # 통계
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            # 진행 상황 업데이트
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct / train_total:.4f}'
            })
        
        # 에폭 평균 손실 및 정확도
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 평가 모드
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # 평가 루프
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'에폭 {epoch+1}/{num_epochs} [검증]')
            for features, labels in pbar:
                features, labels = features.to(device), labels.to(device)
                
                # 순전파
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # 통계
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                # 진행 상황 업데이트
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{val_correct / val_total:.4f}'
                })
        
        # 에폭 평균 손실 및 정확도
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 에폭 요약 (오류율 반영한 알파값 표시)
        current_alpha = criterion.alpha
        if hasattr(criterion, 'fp_rate') and criterion.fp_rate is not None and criterion.fn_rate is not None and criterion.fn_rate > 0:
            error_ratio = criterion.fp_rate / criterion.fn_rate
            if error_ratio > 1:
                adjustment = min(0.1, np.log10(error_ratio) * 0.05)
                current_alpha = min(0.95, criterion.alpha + adjustment)
            elif error_ratio < 1:
                adjustment = max(-0.1, -np.log10(1/error_ratio) * 0.05)
                current_alpha = max(0.7, criterion.alpha + adjustment)
        
        print(f'에폭 {epoch+1}/{num_epochs}: '
              f'학습 손실: {train_loss:.4f}, 학습 정확도: {train_acc:.4f}, '
              f'검증 손실: {val_loss:.4f}, 검증 정확도: {val_acc:.4f}, '
              f'Alpha: {current_alpha:.4f}')
        
        # 최적 모델 저장 및 조기 종료 검사
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
            print(f"  최적 모델 갱신 (검증 손실: {best_val_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"  조기 종료 카운터: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print(f'조기 종료 (에폭 {epoch+1})')
                break
    
    # 최적 모델 상태 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, train_accs, val_accs

# 학습 함수 정의
def train_accident_detector(data_dir=config.DATA_DIR):
    # 시드 설정
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 장치 설정
    device = config.DEVICE
    print(f'사용 장치: {device}')
    
    # 데이터셋 로드
    dataset = AudioDataset(data_dir)
    if len(dataset) == 0:
        print("오류: 데이터셋이 비어 있습니다. 데이터를 확인하세요.")
        utils.prepare_example_data()
        return
    
    print(f'데이터셋 로드 완료: 총 {len(dataset)}개 샘플')
    
    # 클래스별 샘플 수 계산
    class_samples = [0] * config.NUM_CLASSES
    for sample in dataset.samples:
        class_samples[sample['class']] += 1

    print(f'클래스별 샘플 수: {class_samples}')

    # 클래스 불균형 비율 계산
    total_samples = sum(class_samples)
    class_ratio = class_samples[0] / class_samples[1] if class_samples[1] > 0 else 0
    print(f'클래스 불균형 비율 (정상:사고): {class_ratio:.2f}:1')
    
    # Focal Loss 알파 값 계산 - 수정된 로직: 불균형 정도에 따라 더 세분화된 알파값
    focal_alpha = 0.75  # 기본값
    if class_ratio > 10:
        # 매우 불균형이 심한 경우 (10:1 이상)
        focal_alpha = 0.9  # 사고 클래스에 높은 가중치
    elif class_ratio > 5:
        # 불균형이 심한 경우 (5:1 ~ 10:1)
        focal_alpha = 0.85  # 사고 클래스에 높은 가중치
    elif class_ratio > 2:
        # 중간 정도 불균형 (2:1 ~ 5:1)
        focal_alpha = 0.8  # 사고 클래스에 중간 가중치
    else:
        # 경미한 불균형 (2:1 이하)
        focal_alpha = 0.75  # 사고 클래스에 기본 가중치
    
    print(f'Focal Loss alpha 값: {focal_alpha:.4f} (사고 클래스 가중치)')
    
    # 학습/검증 분할
    val_size = int(len(dataset) * config.VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f'학습 세트: {train_size}개 샘플, 검증 세트: {val_size}개 샘플')
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0  # 멀티프로세싱 오류 방지 위해 0으로 설정
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0  # 멀티프로세싱 오류 방지 위해 0으로 설정
    )
    
    # 모델 초기화
    model = CRNN().to(device)
    print(model)
    
    # 손실 함수로 Focal Loss 적용
    criterion = FocalLoss(alpha=focal_alpha, gamma=2.0)
    print(f'Focal Loss 적용 (alpha={focal_alpha}, gamma=2.0)')
    
    # 옵티마이저 설정 - L2 정규화 유지
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # 모델 학습
    print('학습 시작...')
    start_time = time.time()
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=config.EPOCHS, patience=config.PATIENCE
    )
    elapsed_time = time.time() - start_time
    print(f'학습 완료: {elapsed_time:.2f}초 소요')
    
    # 결과 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 모델 저장
    model_path = os.path.join(config.MODEL_DIR, f'crnn_model_focal_{timestamp}.pth')
    utils.save_model(model, model_path)
    
    # 학습 곡선 저장
    curves_path = os.path.join(config.RESULT_DIR, f'learning_curves_{timestamp}.png')
    utils.plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_path=curves_path)
    
    # 전체 데이터셋에서 성능 평가
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=0):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 사고 클래스의 확률
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    
    # 성능 지표 계산 및 저장
    metrics = utils.calculate_metrics(all_labels, all_preds)
    metrics_path = os.path.join(config.RESULT_DIR, f'metrics_{timestamp}.csv')
    utils.save_metrics(metrics, metrics_path)
    
    # 혼동 행렬 시각화 및 저장
    cm_path = os.path.join(config.RESULT_DIR, f'confusion_matrix_{timestamp}.png')
    utils.plot_confusion_matrix(metrics['confusion_matrix'], save_path=cm_path)
    
    # 추가 그래프 생성
    pr_curve_path = os.path.join(config.RESULT_DIR, f'pr_curve_{timestamp}.png')
    utils.plot_precision_recall_curve(all_labels, all_probs, save_path=pr_curve_path)

    f1_curve_path = os.path.join(config.RESULT_DIR, f'f1_curve_{timestamp}.png')
    utils.plot_f1_score(all_labels, all_probs, save_path=f1_curve_path)

    roc_curve_path = os.path.join(config.RESULT_DIR, f'roc_curve_{timestamp}.png')
    utils.plot_roc_curve(all_labels, all_probs, save_path=roc_curve_path)

    # AUC 계산
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(all_labels, all_probs)
    print(f"  - AUC: {auc:.4f}")
    
    # 최적 임계값 계산 (F1 점수 기준)
    # 이 부분은 추가 개선 사항: F1 점수를 최대화하는 임계값을 찾아 config에 반영
    try:
        
        from sklearn.metrics import f1_score
        # F1 점수를 최대화하는 임계값 찾기
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.linspace(0.1, 0.9, 81):  # 0.1부터 0.9까지 0.01 간격
            pred_at_threshold = (np.array(all_probs) >= threshold).astype(int)
            f1 = f1_score(all_labels, pred_at_threshold)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"\n최적 임계값: {best_threshold:.4f} (F1 점수: {best_f1:.4f})")
        print(f"config.py에서 PREDICTION_THRESHOLD를 {best_threshold:.4f}로 변경하는 것을 고려하세요.")
        
        # 최적 임계값으로 예측 결과 재계산
        best_preds = (np.array(all_probs) >= best_threshold).astype(int)
        best_metrics = utils.calculate_metrics(all_labels, best_preds)
        
        print(f"최적 임계값 적용 시 성능:")
        print(f"  - 정확도: {best_metrics['accuracy']:.4f}")
        print(f"  - 정밀도: {best_metrics['precision']:.4f}")
        print(f"  - 재현율: {best_metrics['recall']:.4f}")
        print(f"  - F1 점수: {best_metrics['f1_score']:.4f}")
        
        # 최적 임계값 혼동 행렬 저장
        best_cm_path = os.path.join(config.RESULT_DIR, f'confusion_matrix_optimal_{timestamp}.png')
        utils.plot_confusion_matrix(best_metrics['confusion_matrix'], save_path=best_cm_path)
    except Exception as e:
        print(f"최적 임계값 계산 중 오류: {str(e)}")
    
    print('학습 및 평가 완료!')
    print(f'성능 지표:')
    print(f"  - 정확도: {metrics['accuracy']:.4f}")
    print(f"  - 정밀도: {metrics['precision']:.4f}")
    print(f"  - 재현율: {metrics['recall']:.4f}")
    print(f"  - F1 점수: {metrics['f1_score']:.4f}")
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='사고 감지 모델 학습')
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR, help='데이터 디렉토리 경로')
    
    args = parser.parse_args()
    train_accident_detector(args.data_dir)