import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    """
    Бейслайн модель: всегда предсказывает самый частый класс.
    Автоматически определяет самый частый класс из обучающих данных.
    """
    
    def __init__(self):
        self.classes_ = None
        self.n_features_in_ = None

    
    def fit(self, X, y):
        """
        Обучает модель (определяет самый частый класс).
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Признаки (игнорируются, но проверяются на корректность).
        y : array-like, shape (n_samples,)
            Целевая переменная.
            
        Returns:
        --------
        self : object
        """
        # Проверяем входные данные
        X, y = check_X_y(X, y)
        
        # Определяем уникальные классы
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        
        # определяем самый частый класс
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        # Находим индекс самого частого класса
        majority_idx = np.argmax(class_counts)
        
        # Сохраняем информацию о самом частом классе
        self.majority_class_ = unique_classes[majority_idx]
        self.majority_class_count_ = class_counts[majority_idx]
        self.majority_class_proportion_ = class_counts[majority_idx] / len(y)
        
        # Сохраняем полное распределение классов для predict_proba
        self.class_distribution_ = {}
        self.class_counts_ = {}
        
        for cls, count in zip(unique_classes, class_counts):
            self.class_distribution_[cls] = count / len(y)
            self.class_counts_[cls] = count
        
        # Для отладки и отчётов
        print(f"[MajorityClassifier] Обучение завершено:")
        print(f"  Всего образцов: {len(y)}")
        print(f"  Самый частый класс: {self.majority_class_} "
              f"({self.majority_class_proportion_:.1%}, {self.majority_class_count_} образцов)")
        print(f"  Распределение классов: {self.class_distribution_}")
        
        return self
    
    def predict(self, X):
        """
        Предсказывает самый частый класс для всех образцов.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Признаки (игнорируются).
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Предсказанные метки (всегда самый частый класс).
        """
        # Проверяем, что модель обучена
        check_is_fitted(self, ['majority_class_', 'classes_'])
        
        # Проверяем входные данные
        X = check_array(X)
        
        # Всегда предсказываем самый частый класс
        n_samples = X.shape[0]
        return np.full(n_samples, self.majority_class_, dtype=self.classes_.dtype)
    
    def predict_proba(self, X):
        """
        Возвращает вероятности (распределение классов из обучающей выборки).
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Признаки (игнорируются).
            
        Returns:
        --------
        proba : array, shape (n_samples, n_classes)
            Вероятности классов (одинаковые для всех образцов).
        """
        check_is_fitted(self, ['class_distribution_', 'classes_'])
        X = check_array(X)
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Создаем матрицу вероятностей
        proba = np.zeros((n_samples, n_classes))
        
        # Заполняем одинаковыми вероятностями для всех образцов
        for i, cls in enumerate(self.classes_):
            proba[:, i] = self.class_distribution_[cls]
        
        return proba



