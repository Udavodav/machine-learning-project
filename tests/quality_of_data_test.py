
import pandas as pd
import numpy as np
import pytest

class TestDataQuality:
    
    def test_no_target_leakage(self, get_train_data):
        """Проверка утечки таргета"""
        X_train, y_train = get_train_data
        
        # Проверяем, что в фичах нет таргета
        assert 'Churn' not in X_train
        
        # Проверяем, что нет производных от таргета
        for col in X_train.columns:
            if col != 'Churn':
                # Проверяем корреляцию (не должно быть слишком высокой)
                corr = X_train[col].corr(y_train)
                assert abs(corr) < 0.95, f"Высокая корреляция с таргетом в {col}: {corr}"
    
    
    def test_missing_values_after_processing(self, get_train_data):
        """Проверка пропусков после обработки"""
        X_train, y_train = get_train_data
        
        # Проверяем пропуски в X_train
        missing_in_X = X_train.isnull().sum()
        total_missing_X = missing_in_X.sum()
        
        # Проверяем пропуски в y_train (если это DataFrame)
        if isinstance(y_train, pd.DataFrame):
            missing_in_y = y_train.isnull().sum()
            total_missing_y = missing_in_y.sum()
        else:
            # Если y_train это Series
            total_missing_y = pd.isnull(y_train).sum()
            missing_in_y = pd.Series({'Churn': total_missing_y})
        
        print(f"\nПропуски в X_train:")
        for col, missing_count in missing_in_X.items():
            if missing_count > 0:
                percentage = (missing_count / len(X_train)) * 100
                print(f"  {col}: {missing_count} пропусков ({percentage:.1f}%)")
            else:
                print(f"  {col}: 0 пропусков")
        
        print(f"\nПропуски в y_train:")
        for col, missing_count in missing_in_y.items():
            if missing_count > 0:
                percentage = (missing_count / len(y_train)) * 100
                print(f"  {col}: {missing_count} пропусков ({percentage:.1f}%)")
            else:
                print(f"  {col}: 0 пропусков")
        
        # Assertions
        # В обработанных данных не должно быть пропусков
        assert total_missing_X == 0, f"Найдены {total_missing_X} пропусков в X_train"
        assert total_missing_y == 0, f"Найдены {total_missing_y} пропусков в y_train"
        
        # 4. Дополнительная проверка: бесконечные значения
        infinite_X = np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum()
        infinite_y = np.isinf(y_train.select_dtypes(include=[np.number])).sum().sum() if hasattr(y_train, 'select_dtypes') else 0
        
        assert infinite_X == 0, f"Найдены {infinite_X} бесконечных значений в X_train"
        assert infinite_y == 0, f"Найдены {infinite_y} бесконечных значений в y_train"
        
        print(f"\nВсе проверки пропусков пройдены!")
    
    def test_class_balance_report(self, get_train_data):
        """Проверка отчета о балансе классов"""
        X_train, y_train = get_train_data
        
       
        class_counts = y_train.value_counts()
        class_ratio = class_counts.min() / class_counts.max()
        
        print(f"Баланс классов: {class_counts.to_dict()}")
        print(f"Соотношение: {class_ratio:.2f}")
        
        # Можно добавить assertion
        assert class_counts[0] > 0, "Нет примеров класса 0"
        assert class_counts[1] > 0, "Нет примеров класса 1"