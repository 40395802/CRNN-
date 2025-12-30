import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import time
from datetime import datetime
import pandas as pd

import config
from model import CRNN
from dataset import AudioDataset
from train import train_model, FocalLoss
import utils

def k_fold_cross_validation(data_dir=config.DATA_DIR, k=5):
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
    
    # Focal Loss 알파 값 계산 (더 불균형이 심할수록 사고 클래스에 더 높은 가중치)
    focal_alpha = min(0.9, max(0.75, class_samples[0] / total_samples))
    print(f'Focal Loss alpha 값: {focal_alpha:.4f} (사고 클래스 가중치)')
    
    # K-Fold 교차 검증 설정
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # 각 폴드별 성능 저장
    fold_metrics = []
    
    # 데이터셋 인덱스
    indices = list(range(len(dataset)))
    
    # 타임스탬프 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 각 폴드에서 모델 학습 및 평가
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f'{"="*20} 폴드 {fold+1}/{k} {"="*20}')
        
        # 데이터 로더 생성
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=0
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            sampler=val_sampler,
            num_workers=0
        )
        
        # 폴드에서의 클래스 샘플 수 계산
        fold_class_samples = [0] * config.NUM_CLASSES
        for idx in train_idx:
            sample = dataset.samples[idx]
            fold_class_samples[sample['class']] += 1
        
        fold_total_samples = sum(fold_class_samples)
        
        fold_class_ratio = fold_class_samples[0] / fold_class_samples[1] if fold_class_samples[1] > 0 else 0
        if fold_class_ratio > 10:
            # 매우 불균형이 심한 경우 (10:1 이상)
            fold_focal_alpha = 0.9  # 사고 클래스에 높은 가중치
        elif fold_class_ratio > 5:
            # 불균형이 심한 경우 (5:1 ~ 10:1)
            fold_focal_alpha = 0.85  # 사고 클래스에 높은 가중치
        elif fold_class_ratio > 2:
            # 중간 정도 불균형 (2:1 ~ 5:1)
            fold_focal_alpha = 0.8  # 사고 클래스에 중간 가중치
        else:
            # 경미한 불균형 (2:1 이하)
            fold_focal_alpha = 0.75  # 사고 클래스에 기본 가중치

        # Focal Loss 설정
        criterion = FocalLoss(alpha=fold_focal_alpha, gamma=2.0)

        print(f'폴드 {fold+1} Focal Loss alpha 값: {fold_focal_alpha:.4f} (사고 클래스 가중치)')
        
        # 옵티마이저 설정 - L2 정규화 유지
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 모델 학습
        print(f'폴드 {fold+1} 학습 시작...')
        start_time = time.time()
        model, train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, criterion, optimizer, device,
            num_epochs=config.EPOCHS, patience=config.PATIENCE
        )
        elapsed_time = time.time() - start_time
        print(f'폴드 {fold+1} 학습 완료: {elapsed_time:.2f}초 소요')
        
        # 모델 저장
        model_path = os.path.join(config.MODEL_DIR, f'crnn_model_fold{fold+1}_focal_{timestamp}.pth')
        utils.save_model(model, model_path)
        
        # 학습 곡선 저장
        curves_path = os.path.join(config.RESULT_DIR, f'learning_curves_fold{fold+1}_{timestamp}.png')
        utils.plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_path=curves_path)
        
        # 검증 세트에서 성능 평가
        model.eval()
        val_preds = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for features, labels in DataLoader(
                dataset,
                batch_size=config.BATCH_SIZE,
                sampler=val_sampler,
                num_workers=0
            ):
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs)
        
        # 성능 지표 계산
        metrics = utils.calculate_metrics(val_labels, val_preds)
        
        # 추가 그래프 생성
        pr_curve_path = os.path.join(config.RESULT_DIR, f'pr_curve_fold{fold+1}_{timestamp}.png')
        utils.plot_precision_recall_curve(val_labels, val_probs, save_path=pr_curve_path)
        
        f1_curve_path = os.path.join(config.RESULT_DIR, f'f1_curve_fold{fold+1}_{timestamp}.png')
        utils.plot_f1_score(val_labels, val_probs, save_path=f1_curve_path)
        
        roc_curve_path = os.path.join(config.RESULT_DIR, f'roc_curve_fold{fold+1}_{timestamp}.png')
        utils.plot_roc_curve(val_labels, val_probs, save_path=roc_curve_path)
        
        # AUC 계산
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(val_labels, val_probs)
        metrics['auc'] = auc
        
        # 성능 지표 출력
        print(f'폴드 {fold+1} 성능 지표:')
        print(f'  - 정확도: {metrics["accuracy"]:.4f}')
        print(f'  - 정밀도: {metrics["precision"]:.4f}')
        print(f'  - 재현율: {metrics["recall"]:.4f}')
        print(f'  - F1 점수: {metrics["f1_score"]:.4f}')
        print(f'  - AUC: {metrics["auc"]:.4f}')
        
        # 폴드별 성능 지표 저장
        fold_metrics.append(metrics)
        
        # 혼동 행렬 시각화 및 저장
        cm_path = os.path.join(config.RESULT_DIR, f'confusion_matrix_fold{fold+1}_{timestamp}.png')
        utils.plot_confusion_matrix(metrics['confusion_matrix'], save_path=cm_path)
    
    # 전체 폴드 평균 성능 계산
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in fold_metrics]),
        'auc': np.mean([m['auc'] for m in fold_metrics])
    }
    
    # 표준 편차 계산
    std_metrics = {
        'accuracy_std': np.std([m['accuracy'] for m in fold_metrics]),
        'precision_std': np.std([m['precision'] for m in fold_metrics]),
        'recall_std': np.std([m['recall'] for m in fold_metrics]),
        'f1_score_std': np.std([m['f1_score'] for m in fold_metrics]),
        'auc_std': np.std([m['auc'] for m in fold_metrics])
    }
    
    # 평균 성능 출력
    print(f'\n{"="*20} 교차 검증 결과 {"="*20}')
    print(f'총 {k}개 폴드 평균 성능:')
    print(f'  - 정확도: {avg_metrics["accuracy"]:.4f} ± {std_metrics["accuracy_std"]:.4f}')
    print(f'  - 정밀도: {avg_metrics["precision"]:.4f} ± {std_metrics["precision_std"]:.4f}')
    print(f'  - 재현율: {avg_metrics["recall"]:.4f} ± {std_metrics["recall_std"]:.4f}')
    print(f'  - F1 점수: {avg_metrics["f1_score"]:.4f} ± {std_metrics["f1_score_std"]:.4f}')
    print(f'  - AUC: {avg_metrics["auc"]:.4f} ± {std_metrics["auc_std"]:.4f}')
    
    # 평균 성능 저장
    avg_metrics_path = os.path.join(config.RESULT_DIR, f'cross_validation_metrics_focal_{timestamp}.csv')
    avg_metrics.update(std_metrics)
    avg_metrics_df = pd.DataFrame({k: [v] for k, v in avg_metrics.items()})
    avg_metrics_df.to_csv(avg_metrics_path, index=False)
    print(f"\n교차 검증 결과가 {avg_metrics_path}에 저장되었습니다.")
    
    return fold_metrics, avg_metrics