# tests/test_reproducibility.py
import pytest
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier

def test_reproducibility_with_fixed_seed(sample_train_test_split):
    """Тест воспроизводимости с фиксированным seed"""
    # Правильная распаковка: X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = sample_train_test_split
    
    # Проверяем размеры
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Фиксируем все случайные сиды
    random.seed(42)
    np.random.seed(42)
    
    # Обучаем две одинаковые модели
    model1 = RandomForestClassifier(n_estimators=10, random_state=42)
    model2 = RandomForestClassifier(n_estimators=10, random_state=42)
    
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    
    # Проверяем воспроизводимость
    pred1 = model1.predict(X_test)
    pred2 = model2.predict(X_test)
    proba1 = model1.predict_proba(X_test)
    proba2 = model2.predict_proba(X_test)
    
    # Предсказания должны быть идентичны
    assert np.array_equal(pred1, pred2), "Предсказания должны быть одинаковыми"
    
    # Feature importances должны быть идентичны
    assert np.array_equal(model1.feature_importances_, 
                         model2.feature_importances_), \
           "Feature importances должны быть одинаковыми"
    
    # Вероятности должны быть близки (из-за численной погрешности)
    assert np.allclose(proba1, proba2, rtol=1e-5), "Вероятности не воспроизводимы"

def test_deterministic_pipeline():
    """Тест детерминированности всего пайплайна"""
    # Этот тест нужно адаптировать под ваш полный пайплайн
    # Идея: запустить весь пайплайн дважды с одинаковыми сидами
    # и сравнить результаты
    
    pass

def test_cross_validation_reproducibility(sample_classification_data):
    """Тест воспроизводимости кросс-валидации"""
    X, y = sample_classification_data
    
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    
    # Две одинаковые CV стратегии
    cv1 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    scores1 = cross_val_score(model, X, y, cv=cv1, scoring='roc_auc')
    scores2 = cross_val_score(model, X, y, cv=cv2, scoring='roc_auc')
    
    # Результаты должны совпадать
    assert np.allclose(scores1, scores2), "CV scores не воспроизводимы"