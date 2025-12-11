import pytest
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from notebooks.pckgs.majority import MajorityClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def load_final_model(model_path='models/final_model_balanced_medium.pkl'):
    """Загрузка финальной модели"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена. Проверенные пути: {alt_paths}")
    
    model = joblib.load(model_path)
    print(f" Модель загружена из: {model_path}")
    print(f"   Тип модели: {type(model).__name__}")
    return model

def test_baseline_comparison(get_train_test_data):
    """Сравнение с бейслайном на тестовых данных"""
    X_train, X_test, y_train, y_test = get_train_test_data
    
    # Бейслайн 1
    majority_model = MajorityClassifier()
    majority_model.fit(X_train, y_train)
    baseline_pred = majority_model.predict(X_test)
    baseline_proba = majority_model.predict_proba(X_test)
    baseline_f1 = f1_score(y_test, baseline_pred, zero_division=0)
    
    # Бейслайн 2
    lr_model = LogisticRegression(random_state=42, max_iter=100)
    lr_model.fit(X_train, y_train)
    baseline_random_pred = lr_model.predict(X_test)
    baseline_random_proba = lr_model.predict_proba(X_test)[:, 1]
    baseline_random_f1 = f1_score(y_test, baseline_random_pred, zero_division=0)
    
    # Загружаем финальную модель
    try:
        final_model = load_final_model()
    except FileNotFoundError as e:
        pytest.skip(f"Модель не найдена: {e}")
        return
    
    #  Предсказания финальной модели
    model_pred = final_model.predict(X_test)
    model_proba = final_model.predict_proba(X_test)[:, 1] if hasattr(final_model, 'predict_proba') else None
    
    # 4. Расчет метрик
    model_f1 = f1_score(y_test, model_pred)
    
    print(f"\n📊 Сравнение с бейслайнами:")
    print(f"   F1-Score бейслайн (majority): {baseline_f1:.3f}")
    print(f"   F1-Score бейслайн (random):   {baseline_random_f1:.3f}")
    print(f"   F1-Score финальная модель:    {model_f1:.3f}")
    
    # Проверки
    # Проверяем, что модель лучше  бейслайна
    assert model_f1 > baseline_f1, f"Модель ({model_f1:.3f}) должна быть лучше  бейслайна ({baseline_f1:.3f})"

    assert model_f1 > baseline_random_f1, f"Модель ({model_f1:.3f}) должна быть лучше  бейслайна ({baseline_random_f1:.3f})"
    
    # Проверяем ROC-AUC если есть вероятности
    if model_proba is not None:
        model_roc_auc = roc_auc_score(y_test, model_proba)
        print(f"   ROC-AUC финальная модель:     {model_roc_auc:.3f}")
        assert model_roc_auc > 0.5, f"ROC-AUC {model_roc_auc:.3f} должен быть > 0.5"
    
    print("Тест пройден: модель превосходит бейслайны")

def test_minimum_roc_auc_threshold(get_test_data):
    """Проверка минимального порога ROC-AUC из требований проекта"""
    X_test, y_test = get_test_data
    
    # Загрузка модели
    try:
        model = load_final_model()
    except FileNotFoundError as e:
        pytest.skip(f"Модель не найдена: {e}")
        return
    
    # Проверяем что модель поддерживает вероятности
    if not hasattr(model, 'predict_proba'):
        pytest.skip("Модель не поддерживает predict_proba")
        return
    
    # Предсказания
    y_proba = model.predict_proba(X_test)[:, 1]
    model_roc_auc = roc_auc_score(y_test, y_proba)
    
    # Минимальное требование из проекта
    MIN_ROC_AUC = 0.75
    
    print(f"\nПроверка минимального ROC-AUC ({MIN_ROC_AUC}):")
    print(f"   ROC-AUC модели: {model_roc_auc:.3f}")
    
    assert model_roc_auc >= MIN_ROC_AUC, \
        f"ROC-AUC модели ({model_roc_auc:.3f}) ниже минимального требования ({MIN_ROC_AUC})"
    
    print(f"Модель соответствует минимальному требованию ROC-AUC")

def test_overfitting_check(get_train_test_data):
    """Проверка на переобучение на реальных данных"""
    X_train, X_test, y_train, y_test = get_train_test_data
    
    # Загрузка модели
    try:
        model = load_final_model()
    except FileNotFoundError as e:
        pytest.skip(f"Модель не найдена: {e}")
        return
   
    # Оценка загруженной модели
    train_pred_final = model.predict(X_train) if len(X_train) > 0 else []
    test_pred_final = model.predict(X_test)
    
    train_accuracy_final = accuracy_score(y_train, train_pred_final)
    test_accuracy_final = accuracy_score(y_test, test_pred_final)
    accuracy_gap_final = train_accuracy_final - test_accuracy_final

    # Проверяем, что model не слишком переобучен
    assert accuracy_gap_final < 0.3, \
        f"Слишком большой разрыв у model: {accuracy_gap_final:.3f}"
    
    print("Тест на переобучение пройден")

def test_model_loading():
    """Тест загрузки модели"""
    print("\nТест загрузки модели...")
    
    try:
        model = load_final_model()
        
        # Проверяем базовые методы
        assert hasattr(model, 'predict'), "Модель должна иметь метод predict"
        
        if hasattr(model, 'predict_proba'):
            print("   ✓ Модель поддерживает predict_proba")
        
        # Проверяем атрибуты модели
        print(f"   Модель загружена успешно")
        print(f"   Тип: {type(model).__name__}")
        
        # Попробуем сделать dummy prediction
        dummy_X = np.zeros((1, 10))  # 1 sample, 10 features
        try:
            pred = model.predict(dummy_X)
            print(f"   Модель может делать предсказания")
        except Exception as e:
            print(f"   ⚠Предсказание не удалось: {e}")
        
    except FileNotFoundError as e:
        pytest.fail(f"Не удалось загрузить модель: {e}")
    
    print("Тест загрузки модели пройден")