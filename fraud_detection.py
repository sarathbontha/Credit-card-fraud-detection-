"""Unsupervised credit-card anomaly detection with an audit by transaction amount.

Dataset: https://www.kaggle.com/datasets/whenamancodes/fraud-detection
Run: python fraud_detection.py --data creditcard.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURES = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']


def load_transactions(path: Path) -> pd.DataFrame:
    """Read and validate the expected Kaggle credit-card dataset."""
    if not path.is_file():
        raise FileNotFoundError(f'Dataset not found: {path}. Download creditcard.csv from Kaggle.')
    frame = pd.read_csv(path)
    required = set(FEATURES + ['Class'])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    frame = frame[FEATURES + ['Class']].copy()
    if frame.isna().any().any():
        raise ValueError('Dataset has missing values; review them before training.')
    if not frame['Class'].isin([0, 1]).all():
        raise ValueError('Class must contain only 0 (normal) or 1 (fraud).')
    if frame['Class'].nunique() != 2:
        raise ValueError('Both normal and fraud labels are needed for evaluation.')
    return frame


def run(data_path: Path, output_dir: Path, epochs: int, max_train: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_transactions(data_path)
    # Labels are reserved for holdout evaluation, not for AutoEncoder training.
    development, test = train_test_split(frame, test_size=0.20,
                                         stratify=frame['Class'], random_state=SEED)
    normal_development = development.loc[development['Class'] == 0]
    normal_train, normal_calibration = train_test_split(
        normal_development, test_size=0.15, random_state=SEED)
    if len(normal_train) > max_train:
        normal_train = normal_train.sample(n=max_train, random_state=SEED)

    # Log Amount to reduce skew; fit transforms only on normal training rows.
    def prepare(part: pd.DataFrame) -> pd.DataFrame:
        features = part[FEATURES].copy()
        features['Amount'] = np.log1p(features['Amount'].clip(lower=0))
        return features

    scaler = StandardScaler()
    x_train = scaler.fit_transform(prepare(normal_train))
    x_calibration = scaler.transform(prepare(normal_calibration))
    x_test = scaler.transform(prepare(test))

    # PyOD computes reconstruction-based anomaly scores; higher means stranger.
    model = AutoEncoder(hidden_neuron_list=[32, 16, 8, 16, 32],
                        epoch_num=epochs, batch_size=256, contamination=0.005,
                        preprocessing=False, random_state=SEED, verbose=0)
    model.fit(x_train)
    calibration_scores = model.decision_function(x_calibration)
    test_scores = model.decision_function(x_test)
    # Set a threshold without looking at the test labels.
    threshold = float(np.quantile(calibration_scores, 0.995))
    predicted = (test_scores > threshold).astype(int)
    truth = test['Class'].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()

    results = {
        'dataset_rows': int(len(frame)),
        'normal_training_rows': int(len(normal_train)),
        'normal_calibration_rows': int(len(normal_calibration)),
        'test_rows': int(len(test)),
        'fraud_in_test': int(truth.sum()),
        'threshold': threshold,
        'precision': float(precision_score(truth, predicted, zero_division=0)),
        'recall': float(recall_score(truth, predicted, zero_division=0)),
        'average_precision': float(average_precision_score(truth, test_scores)),
        'roc_auc': float(roc_auc_score(truth, test_scores)),
        'true_positive': int(tp), 'false_positive': int(fp),
        'true_negative': int(tn), 'false_negative': int(fn),
        'amount_audit': {},
    }
    # Additional original analysis: compare recall across transaction sizes.
    for label, mask in {
        'under_50': test['Amount'].to_numpy() < 50,
        '50_to_500': (test['Amount'].to_numpy() >= 50) & (test['Amount'].to_numpy() < 500),
        '500_and_above': test['Amount'].to_numpy() >= 500,
    }.items():
        group_truth = truth[mask]
        group_pred = predicted[mask]
        fraud_count = int(group_truth.sum())
        results['amount_audit'][label] = {
            'transactions': int(mask.sum()), 'fraud_cases': fraud_count,
            'fraud_detected': int(((group_truth == 1) & (group_pred == 1)).sum()),
            'fraud_recall': (float(recall_score(group_truth, group_pred, zero_division=0))
                             if fraud_count else None),
        }

    (output_dir / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    pd.DataFrame({'actual_class': truth, 'anomaly_score': test_scores,
                  'predicted_class': predicted, 'amount': test['Amount'].to_numpy()}
                 ).to_csv(output_dir / 'test_predictions.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, name in [(0, 'Normal'), (1, 'Fraud')]:
        values = test_scores[truth == label]
        ax.hist(values, bins=50, alpha=0.5, label=f'{name} (n={len(values)})')
    ax.axvline(threshold, color='black', linestyle='--', label='Calibration threshold')
    ax.set(xlabel='Reconstruction-based anomaly score', ylabel='Transactions',
           title='AutoEncoder anomaly scores on held-out test data')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'score_distribution.png', dpi=160)
    plt.close(fig)

    print('FRAUD DETECTION EXPERIMENT RESULTS')
    print(f'Test transactions: {len(test):,}; fraud cases: {truth.sum():,}')
    print(f'Precision: {results["precision"]:.4f} | Recall: {results["recall"]:.4f}')
    print(f'Average precision: {results["average_precision"]:.4f} | ROC AUC: {results["roc_auc"]:.4f}')
    print(f'TP={tp}, FP={fp}, TN={tn}, FN={fn}')
    print('Transaction amount audit:')
    for name, audit in results['amount_audit'].items():
        print(f'  {name}: {audit}')
    print(f'Files saved to: {output_dir.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and audit a PyOD AutoEncoder.')
    parser.add_argument('--data', type=Path, default=Path('creditcard.csv'))
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--max-train', type=int, default=20000)
    args = parser.parse_args()
    if args.epochs < 1 or args.max_train < 100:
        parser.error('--epochs must be positive and --max-train must be at least 100')
    run(args.data, args.output, args.epochs, args.max_train)
