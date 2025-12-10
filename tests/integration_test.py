
import pytest
import numpy as np
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append('..')  # поднимаемся из tests/ на уровень выше
# Импортируем
from notebooks.models import *

def test_calibrate_model_integration(sample_train_test_split):
    """Integration тест функции калибровки"""
    X_train, X_test, y_train, y_test = sample_train_test_split
    
    # Создаем простую модель
    base_model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Разделяем train на train и val
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Тестируем функцию калибровки
    final_model, best_method, calibration_results = calibrate_model(
        base_model, X_train_split, y_train_split, X_val, y_val, X_test, y_test
    )
    
    # Проверяем возвращаемые значения
    assert final_model is not None
    assert isinstance(best_method, str)
    assert best_method in ['Без калибровки', 'Platt scaling', 'Isotonic regression']
    
    # Проверяем структуру результатов
    assert 'ROC-AUC' in calibration_results.columns
    assert 'PR-AUC' in calibration_results.columns
    assert 'Brier Score' in calibration_results.columns
    
    # Проверяем, что финальная модель может делать предсказания
    predictions = final_model.predict(X_test)
    probabilities = final_model.predict_proba(X_test)
    
    assert len(predictions) == len(X_test)
    assert probabilities.shape == (len(X_test), 2)
    assert (probabilities >= 0).all() and (probabilities <= 1).all()

def test_ensemble_creation_functions():
    """Тест функций создания ансамблей"""
    from sklearn.linear_model import LogisticRegression
    
    # Тестируем Voting ансамбль
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=10)),
        ('lr', LogisticRegression())
    ]
    
    voting_ensemble = create_voting_ensemble(estimators)
    assert voting_ensemble is not None
    assert len(voting_ensemble.estimators) == 2
    
    # Тестируем Stacking ансамбль
    stacking_ensemble = create_stacking_ensemble(
        estimators, LogisticRegression(), cv=2
    )
    assert stacking_ensemble is not None
    assert stacking_ensemble.final_estimator is not None

def test_model_saving_and_loading(tmp_path):
    """Тест сохранения и загрузки моделей"""
    # Создаем простую модель
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Мокаем данные для обучения
    X = np.random.randn(10, 5)
    y = np.random.randint(0, 2, 10)
    model.fit(X, y)
    
    # Сохраняем модель
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(model, model_path)
    
    # Загружаем модель
    loaded_model = joblib.load(model_path)
    
    # Проверяем, что модель загрузилась
    assert loaded_model is not None
    
    # Проверяем, что может делать предсказания
    X_test = np.random.randn(5, 5)
    predictions = loaded_model.predict(X_test)
    assert len(predictions) == 5