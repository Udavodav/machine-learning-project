# tests/test_models.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

import sys

sys.path.append('..')  # поднимаемся из tests/ на уровень выше

# Импортируем
from notebooks.models import *

# from models import (randForestOpt, gradOpt, calculate_metrics, print_metrics, load_data, split_data_for_calibration)

def test_randForestOpt_function(sample_classification_data):
    """Тест функции оптимизации Random Forest"""
    X, y = sample_classification_data
    
    # Создаем mock trial
    mock_trial = Mock()
    mock_trial.suggest_int = Mock(side_effect=lambda name, low, high: (low + high) // 2)
    mock_trial.suggest_categorical = Mock(side_effect=lambda name, choices: choices[0])
    mock_trial.suggest_float = Mock(return_value=0.05)
    
    # Вызываем функцию
    score = randForestOpt(mock_trial, X, y)
    
    # Проверяем результат
    assert isinstance(score, float)
    assert 0 <= score <= 1  # ROC-AUC должен быть в диапазоне [0, 1]
    
    # Проверяем, что были вызовы suggest методов
    assert mock_trial.suggest_int.call_count >= 4
    assert mock_trial.suggest_categorical.call_count >= 2

def test_gradOpt_function(sample_classification_data):
    """Тест функции оптимизации Gradient Boosting"""
    X, y = sample_classification_data
    
    mock_trial = Mock()
    mock_trial.suggest_int = Mock(side_effect=lambda name, low, high: (low + high) // 2)
    mock_trial.suggest_categorical = Mock(side_effect=lambda name, choices: choices[0])
    mock_trial.suggest_float = Mock(side_effect=[
        100,  # n_estimators (будет проигнорирован из-за side_effect)
        0.05, # learning_rate
        5,    # max_depth
        30,   # min_samples_split
        12,   # min_samples_leaf
        'sqrt', # max_features
        0.7   # subsample
    ])
    
    score = gradOpt(mock_trial, X, y)
    
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_calculate_metrics_function():
    """Тест функции расчета метрик"""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_proba = np.array([0.1, 0.9, 0.6, 0.8, 0.2, 0.4])
    cv_scores = np.array([0.85, 0.90, 0.88])
    
    metrics = calculate_metrics('test_model', y_true, y_pred, y_proba, cv_scores)
    
    # Проверяем наличие всех ключей
    required_keys = ['model', 'test_roc_auc', 'test_pr_auc', 'test_f1', 
                    'test_precision', 'test_recall', 'test_accuracy',
                    'cv_pr_auc_mean', 'cv_pr_auc_std']
    
    for key in required_keys:
        assert key in metrics
    
    # Проверяем типы значений
    assert isinstance(metrics['test_roc_auc'], float)
    assert isinstance(metrics['test_f1'], float)
    assert isinstance(metrics['cv_pr_auc_mean'], float)
    
    # Проверяем корректные диапазоны
    assert 0 <= metrics['test_roc_auc'] <= 1
    assert 0 <= metrics['test_f1'] <= 1
    assert 0 <= metrics['test_accuracy'] <= 1

def test_print_metrics_function(capsys):
    """Тест функции печати метрик"""
    metrics = {
        'test_roc_auc': 0.85,
        'test_pr_auc': 0.80,
        'test_f1': 0.75,
        'test_precision': 0.70,
        'test_recall': 0.80,
        'test_accuracy': 0.85
    }
    
    print_metrics(metrics)
    
    captured = capsys.readouterr()
    output = captured.out
    
    # Проверяем, что все метрики напечатаны
    assert "ROC-AUC" in output
    assert "0.85" in output
    assert "F1-Score" in output
    assert "0.75" in output

def test_load_data_function(mock_csv_data):
    """Тест функции загрузки данных"""
    train_path, test_path = mock_csv_data
    
    # Просто передаем пути в функцию
    X_train, y_train, X_test, y_test = load_data(
        type_data='balanced',
        train_path=train_path,
        test_path=test_path
    )
    
    # Проверки
    assert len(X_train) == 5
    assert len(X_test) == 3

def test_split_data_for_calibration(sample_classification_data):
    """Тест функции разделения данных для калибровки"""
    X, y = sample_classification_data
    
    X_train, X_val, y_train, y_val = split_data_for_calibration(X, y, test_size=0.2)
    
    # Проверяем размеры
    assert len(X_train) + len(X_val) == len(X)
    assert len(y_train) + len(y_val) == len(y)
    
    # Проверяем, что val выборка составляет примерно 20%
    assert abs(len(X_val) / len(X) - 0.2) < 0.01
    
    # Проверяем стратификацию (баланс классов должен сохраняться)
    train_ratio = y_train.mean()
    val_ratio = y_val.mean()
    assert abs(train_ratio - val_ratio) < 0.1



