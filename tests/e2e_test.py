
import pytest
import pandas as pd
import numpy as np
import joblib
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append('..')
from notebooks.models import *

def test_end_to_end_pipeline_minimal(capsys):
    """Минимальный End-to-End тест"""  

    X_train, y_train, X_test, y_test = load_data(type_data="balanced_medium", train_path="data/processed/train_balanced_medium.csv", test_path="data/processed/test.csv")
    
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert 'Churn' not in X_train.columns
    assert 'Churn' not in X_test.columns

    model = joblib.load('models/final_model_balanced_medium.pkl')

    assert hasattr(model, 'predict'), "Модель должна иметь метод predict"

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    assert len(y_pred) == len(y_test)
    assert len(y_proba) == len(y_test)
    assert (y_proba >= 0).all() and (y_proba <= 1).all()

    metrics = calculate_metrics('test_model', y_test, y_pred, y_proba)

    required_metrics = ['test_roc_auc', 'test_pr_auc', 'test_f1', 
                      'test_precision', 'test_recall', 'test_accuracy']
    
    for metric in required_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], float)

    print_metrics(metrics)

    captured = capsys.readouterr()
    output = captured.out
    
    # Проверяем, что все метрики напечатаны
    assert "ROC-AUC" in output
    assert "F1-Score" in output
    assert metrics['test_roc_auc'] > 0.75, f"ROC-AUC слишком низкий: {metrics['test_roc_auc']:.3f}"




