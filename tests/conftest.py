# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


@pytest.fixture
def get_train_data():
    """
    Фикстура возвращает данные для теста модели
    """
    file_train_path = 'data/processed/train_balanced_medium.csv'
    
    if not os.path.exists(file_train_path):
        raise FileNotFoundError(f"Данные не найдены")
        
    df_train = pd.read_csv(file_train_path)
    
    return df_train.drop('Churn', axis=1), df_train['Churn']

@pytest.fixture
def get_train_test_data():
    """
    Фикстура возвращает данные для теста модели
    """
    file_test_path = 'data/processed/test.csv'
    file_train_path = 'data/processed/train_balanced_medium.csv'
    
    if not os.path.exists(file_test_path):
        raise FileNotFoundError(f"Данные не найдены")

    if not os.path.exists(file_train_path):
        raise FileNotFoundError(f"Данные не найдены")
        
    df_test = pd.read_csv(file_test_path)
    df_train = pd.read_csv(file_train_path)
    
    return df_train.drop('Churn', axis=1), df_test.drop('Churn', axis=1), df_train['Churn'], df_test['Churn']

@pytest.fixture
def get_test_data():
    """
    Фикстура возвращает данные для теста модели
    """
    file_path = 'data/processed/test.csv'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Данные не найдены")
        
    df_test = pd.read_csv(file_path)
    
    return df_test.drop('Churn', axis=1), df_test['Churn']

@pytest.fixture
def sample_train_test_split():
    """
    Фикстура возвращает уже разделенные данные для тестов воспроизводимости
    Возвращает: (X_train, X_test, y_train, y_test)
    """
    np.random.seed(42)
    
    # Создаем синтетические данные
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)  # бинарная классификация
    
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # сохраняем распределение классов
    )
    
    return X_train, X_test, y_train, y_test

@pytest.fixture
def sample_classification_data():
    """
    Фикстура создает тестовые данные для классификации
    """
    # Создаем синтетические данные
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)  # бинарная классификация
    
    return X, y

@pytest.fixture
def sample_regression_data():
    """
    Фикстура создает тестовые данные для регрессии
    """
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    return X, y

@pytest.fixture
def mock_csv_data(tmp_path):
    """
    Фикстура создает ДВА временных CSV файла с колонкой 'Churn'
    """
    
    train_data = pd.DataFrame({
        'customerID': ['0001', '0002', '0003', '0004', '0005'],
        'tenure': [12, 24, 36, 48, 60],
        'MonthlyCharges': [70.0, 80.0, 90.0, 100.0, 110.0],
        'TotalCharges': [840.0, 1920.0, 3240.0, 4800.0, 6600.0],
        'InternetService': ['Fiber optic', 'DSL', 'Fiber optic', 'DSL', 'Fiber optic'],
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'],
        'Churn': [0, 1, 0, 1, 0]
    })
    
    
    test_data = pd.DataFrame({
        'customerID': ['0006', '0007', '0008'],
        'tenure': [6, 18, 30],
        'MonthlyCharges': [60.0, 75.0, 85.0],
        'TotalCharges': [360.0, 1350.0, 2550.0],
        'InternetService': ['DSL', 'Fiber optic', 'DSL'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'Churn': [1, 0, 1]
    })
    
    # Сохраняем в файлы
    train_file = tmp_path / "train_data.csv"
    test_file = tmp_path / "test_data.csv"
    
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    
    return str(train_file), str(test_file) 

@pytest.fixture
def sample_dataframe():
    """
    Фикстура создает тестовый DataFrame
    """
    return pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'salary': [50000, 60000, 70000, 80000, 90000],
        'department': ['IT', 'HR', 'IT', 'Sales', 'HR'],
        'churn': [0, 1, 0, 1, 0]
    })