import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score, 
                           accuracy_score, confusion_matrix, roc_curve, average_precision_score)
from sklearn.calibration import CalibratedClassifierCV
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb
from catboost import CatBoostClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from notebooks.pckgs.analyze import *


# Константы
RANDOM_STATE = 42
N_FOLDS = 5
SCORING = 'average_precision'

optuna.logging.set_verbosity(optuna.logging.WARNING) 

# =========== ФУНКЦИИ ДЛЯ ОПТИМИЗАЦИИ ===========
def randForestOpt(trial, X_train, y_train, use_smote=True, sampling_strategy=None):
    """функция для оптимизации RandomForest"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 30),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': True,
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
    }

    pipeline_steps = []
    if use_smote:
        pipeline_steps.append(('smote', SMOTE(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy)))
        
    model = RandomForestClassifier(**params, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    pipeline_steps.append(('classifier', model))
    
    cv_scores = cross_val_score(
        ImbPipeline(pipeline_steps), 
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring=SCORING,
        n_jobs=-1
    )
    
    return cv_scores.mean()

def gradOpt(trial, X_train, y_train, use_smote=True, sampling_strategy=None):
    """функция для оптимизации GradientBoosting"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_samples_split': trial.suggest_int('min_samples_split', 10, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 4, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
    }

    pipeline_steps = []
    if use_smote:
        pipeline_steps.append(('smote', SMOTE(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy)))
        
    model = GradientBoostingClassifier(**params, random_state=RANDOM_STATE)
    pipeline_steps.append(('classifier', model))
    
    cv_scores = cross_val_score(
        ImbPipeline(pipeline_steps), 
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring=SCORING,
        n_jobs=-1
    )
    
    return cv_scores.mean()


def lightgbmOpt(trial, X_train, y_train, use_smote=True, sampling_strategy=None):
    """функция для оптимизации LightGBM"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'random_state': RANDOM_STATE,
        'n_jobs': 10,
        'verbose': -1,  # отключаем логи
    }

    pipeline_steps = []
    if use_smote:
        pipeline_steps.append(('smote', SMOTE(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy)))
        
    model = lgb.LGBMClassifier(**params)
    pipeline_steps.append(('classifier', model))
    
    cv_scores = cross_val_score(
        ImbPipeline(pipeline_steps), 
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring=SCORING,
        n_jobs=10
    )
    
    return cv_scores.mean()

def catboostOpt(trial, X_train, y_train, use_smote=True, sampling_strategy=None):
    """функция для оптимизации CatBoost"""
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 1.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_seed': RANDOM_STATE,
        'verbose': False,  # отключаем логи
    }

    pipeline_steps = []
    if use_smote:
        pipeline_steps.append(('smote', SMOTE(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy)))
        
    model = CatBoostClassifier(**params)
    pipeline_steps.append(('classifier', model))
    
    cv_scores = cross_val_score(
        ImbPipeline(pipeline_steps), 
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring=SCORING,
        n_jobs=1
    )
    
    return cv_scores.mean()


# =========== ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ДАННЫХ ===========
def load_data():
    """
    Загрузка данных для обучения
    """
   
    X_train_bal = pd.read_csv("../data/processed/train_unbalanced.csv").drop('Churn', axis=1)
    y_train_bal = pd.read_csv("../data/processed/train_unbalanced.csv")['Churn']
    X_test_bal = pd.read_csv("../data/processed/test.csv").drop('Churn', axis=1)
    y_test_bal = pd.read_csv("../data/processed/test.csv")['Churn']
    
    return X_train_bal, y_train_bal, X_test_bal, y_test_bal

def split_data_for_calibration(X_train_bal, y_train_bal, test_size=0.2, random_state=42):
    """Разделение данных для калибровки"""
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_bal, y_train_bal, 
        test_size=test_size, 
        stratify=y_train_bal,
        random_state=random_state
    )
    return X_train, X_val, y_train, y_val

# =========== ФУНКЦИИ ДЛЯ ОБУЧЕНИЯ И ОЦЕНКИ ===========
def optimize_hyperparameters(X_train, y_train, optimize_func, n_trials=50, use_smote=True, sampling_strategy=None):
    """Оптимизация гиперпараметров для моделей"""
    print("Подбор параметров...")
    study_rf = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    study_rf.optimize(lambda trial: optimize_func(trial, X_train, y_train, use_smote, sampling_strategy), 
                      n_trials=n_trials, show_progress_bar=True)
    
    return study_rf.best_params

def train_and_evaluate_models(models_dict, X_train, y_train, X_test, y_test, use_smote=True, sampling_strategy=None):
    """Обучение и оценка нескольких моделей"""
    
    cv_strategy = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    results = []
    
    for model_name, model in models_dict.items():
        
        print(f"\n{model_name.upper()}")
        print("-" * 40)

        pipeline_steps = []
        if use_smote:
            pipeline_steps.append(('smote', SMOTE(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy)))
        
        pipeline_steps.append(('classifier', model))
        pipeline = ImbPipeline(pipeline_steps)
        
        # Кросс-валидация
        print("Кросс-валидация (5-fold stratified)")
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv_strategy,
            scoring=SCORING,
            n_jobs=-1
        )
        print(f"   PR-AUC на каждом фолде: {cv_scores}")
        print(f"   Средний PR-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Обучение модели
        if use_smote:
            smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            model.fit(X_train_resampled, y_train_resampled)
        else:
            model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Расчет метрик
        metrics = calculate_metrics(model_name, y_test, y_pred, y_proba, cv_scores)
        results.append(metrics)
        
        # Визуализация
        postfix = model_name.lower().replace(' ', '_')
        rocPlot(y_test, y_proba, f"{type_data}_{postfix}")
        erorrMatrixPlot(y_test, y_proba, f"{type_data}_{postfix}")
        
        print_metrics(metrics)
        
        # Сохраняем модель для использования в калибровке
        models_dict[model_name] = model
    
    return pd.DataFrame(results), models_dict

def calculate_metrics(model_name, y_true, y_pred, y_proba, cv_scores=None):
    """Расчет метрик для модели"""
    metrics = {
        'model': model_name,
        'test_roc_auc': roc_auc_score(y_true, y_proba),
        'test_pr_auc': average_precision_score(y_true, y_proba),
        'test_f1': f1_score(y_true, y_pred),
        'test_precision': precision_score(y_true, y_pred),
        'test_recall': recall_score(y_true, y_pred),
        'test_accuracy': accuracy_score(y_true, y_pred)
    }
    
    if cv_scores is not None:
        metrics['cv_pr_auc_mean'] = cv_scores.mean()
        metrics['cv_pr_auc_std'] = cv_scores.std()
    
    return metrics

def print_metrics(metrics):
    """Печать метрик модели"""
    print(f"Метрики на тесте:")
    print(f"   ROC-AUC:     {metrics['test_roc_auc']:.3f}")
    print(f"   PR-AUC:      {metrics['test_pr_auc']:.3f}")
    print(f"   F1-Score:    {metrics['test_f1']:.3f}")
    print(f"   Precision:   {metrics['test_precision']:.3f}")
    print(f"   Recall:      {metrics['test_recall']:.3f}")
    print(f"   Accuracy:    {metrics['test_accuracy']:.3f}")

# =========== ФУНКЦИИ ДЛЯ КАЛИБРОВКИ ===========
def calibrate_model(base_model, X_train, y_train, X_val, y_val, X_test, y_test, use_smote=True, sampling_strategy=None):
    """Калибровка модели разными методами"""
    # Обучаем базовую модель
    if use_smote:
        smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        base_model.fit(X_train_resampled, y_train_resampled)
    else:
        base_model.fit(X_train, y_train)
    
    # Platt scaling (SIGMOID CALIBRATION)
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method='sigmoid',
        cv='prefit'
    )
    calibrated_model.fit(X_val, y_val)
    
    # Изотоническая регрессия
    isotonic_model = CalibratedClassifierCV(
        base_model,
        method='isotonic',
        cv='prefit'
    )
    isotonic_model.fit(X_val, y_val)
    
    # Оценка калибровки
    print("\nОценка калибровки на тесте:")
    
    # Предсказания разных моделей
    y_proba_base = base_model.predict_proba(X_test)[:, 1]
    y_proba_sigmoid = calibrated_model.predict_proba(X_test)[:, 1]
    y_proba_isotonic = isotonic_model.predict_proba(X_test)[:, 1]
    
    calibration_results = pd.DataFrame({
        'Model': ['Без калибровки', 'Platt scaling', 'Isotonic regression'],
        'ROC-AUC': [
            roc_auc_score(y_test, y_proba_base),
            roc_auc_score(y_test, y_proba_sigmoid),
            roc_auc_score(y_test, y_proba_isotonic)
        ],
        'PR-AUC': [
            average_precision_score(y_test, y_proba_base),
            average_precision_score(y_test, y_proba_sigmoid),
            average_precision_score(y_test, y_proba_isotonic)
        ],
        'Brier Score': [
            np.mean((y_proba_base - y_test) ** 2),
            np.mean((y_proba_sigmoid - y_test) ** 2),
            np.mean((y_proba_isotonic - y_test) ** 2)
        ]
    })
    
    print(calibration_results.to_string(index=False))
    
    # Выбираем лучший калиброванный вариант
    best_calibrated_idx = calibration_results['PR-AUC'].idxmax()
    best_calibrated_model_name = calibration_results.loc[best_calibrated_idx, 'Model']
    
    if best_calibrated_model_name == 'Platt scaling':
        final_model = calibrated_model
    elif best_calibrated_model_name == 'Isotonic regression':
        final_model = isotonic_model
    else:
        final_model = base_model
    
    return final_model, best_calibrated_model_name, calibration_results

# =========== ФУНКЦИИ ДЛЯ АНСАМБЛЕЙ ===========
def create_voting_ensemble(estimators, voting='soft', weights=None):
    """Создание Voting ансамбля"""
    return VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=weights,
        n_jobs=-1
    )

def create_stacking_ensemble(estimators, final_estimator, cv=3):
    """Создание Stacking ансамбля"""
    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,
        passthrough=False,
        n_jobs=-1
    )

def evaluate_ensemble(ensemble, X_train, y_train, X_test, y_test, ensemble_name, use_smote=True, sampling_strategy=None):
    """Оценка ансамбля"""
    print(f"Обучение ансамбля {ensemble_name}...")
    
    if use_smote:
        smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        ensemble.fit(X_train_resampled, y_train_resampled)
    else:
        ensemble.fit(X_train, y_train)
    
    # Предсказания
    ensemble_proba = ensemble.predict_proba(X_test)[:, 1]
    ensemble_pred = ensemble.predict(X_test)
    
    # Визуализация
    rocPlot(y_test, ensemble_proba, f"{ensemble_name}_{type_data}")
    erorrMatrixPlot(y_test, ensemble_proba, f"{ensemble_name}_{type_data}")
    
    # Метрики
    metrics = calculate_metrics(
        ensemble_name, 
        y_test, 
        ensemble_pred, 
        ensemble_proba
    )
    
    print(f"\n Результаты ансамбля {ensemble_name}:")
    print_metrics(metrics)
    
    return ensemble, metrics

# =========== ФУНКЦИИ ДЛЯ СОХРАНЕНИЯ ===========
def save_model(model, filepath):
    """Сохранение модели"""
    joblib.dump(model, filepath)
    print(f"Модель сохранена в {filepath}")


# =========== ОСНОВНОЙ ПАЙПЛАЙН ===========
def run_pipeline(type_data='balanced_medium', sampling_strategy=None):
    """Основной пайплайн обучения"""
    print(f"Запуск пайплайна для данных: {type_data}")
    print("=" * 50)
    
    # Загрузка данных
    X_train_bal, y_train_bal, X_test, y_test = load_data()
    
    print(f"Размеры данных:")
    print(f"  Обучающая (сбалансированная): {X_train_bal.shape}")
    print(f"  Тестовая: {X_test.shape}")
    
    # Разделение для калибровки
    X_train, X_val, y_train, y_val = split_data_for_calibration(X_train_bal, y_train_bal)
    
    print(f"  Train для обучения: {X_train.shape}")
    print(f"  Val для калибровки: {X_val.shape}")

    use_smote = sampling_strategy is not None
    
    print("\nОптимизация RandomForest...")
    forest_params = optimize_hyperparameters(X_train_bal, y_train_bal, randForestOpt, 50, use_smote, sampling_strategy)
    
    print("\nОптимизация GradientBoosting...")
    grad_params = optimize_hyperparameters(X_train_bal, y_train_bal, gradOpt, 50, use_smote, sampling_strategy)
    
    print("\nОптимизация LightGBM...")
    lgb_params = optimize_hyperparameters(X_train_bal, y_train_bal, lightgbmOpt, 30, use_smote, sampling_strategy)
    
    print("\nОптимизация CatBoost...")
    cat_params = optimize_hyperparameters(X_train_bal, y_train_bal, catboostOpt, 15, use_smote, sampling_strategy)
    
    models = {
        'Random Forest': RandomForestClassifier(
            **forest_params,
            class_weight= 'balanced' if use_smote else None,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(**grad_params, random_state=RANDOM_STATE),
        'LightGBM': lgb.LGBMClassifier(**lgb_params, verbose=-1, n_jobs=-1, class_weight='balanced' if use_smote else None),
        'CatBoost': CatBoostClassifier(**cat_params, verbose=False, auto_class_weights='Balanced' if use_smote else None)
    }
    
    results_df, trained_models = train_and_evaluate_models(models, X_train, y_train, X_test, y_test, use_smote, sampling_strategy)
    
    # Выбор лучшей модели
    best_model_info = results_df.loc[results_df['test_pr_auc'].idxmax()]
    best_model_name = best_model_info['model']
    
    print(f"\nЛучшая модель по PR-AUC: {best_model_name}")
    print(f"   PR-AUC на тесте: {best_model_info['test_pr_auc']:.3f}")
    print(f"   PR-AUC на кросс-валидации: {best_model_info['cv_pr_auc_mean']:.3f}")
    
    # Калибровка лучшей модели
    print(f"\nКАЛИБРОВКА МОДЕЛИ {best_model_name}")
    
    
    if best_model_name == 'Random Forest':
        base_model = RandomForestClassifier(
            **forest_params,
            class_weight='balanced' if use_smote else None, 
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    elif best_model_name == 'Gradient Boosting':
        base_model = GradientBoostingClassifier(
            **grad_params,
            random_state=RANDOM_STATE
        )
    elif best_model_name == 'LightGBM':
        base_model = lgb.LGBMClassifier(**lgb_params, n_jobs=-1, class_weight='balanced' if use_smote else None,)
    elif best_model_name == 'CatBoost':
        base_model = CatBoostClassifier(**cat_params, auto_class_weights='Balanced' if use_smote else None)
    
    
    final_model, best_calibration_method, calibration_results = calibrate_model(base_model, X_train, y_train, X_val, y_val, X_test, y_test, use_smote, sampling_strategy)
    
    print(f"\nФинальная модель: {best_model_name} с {best_calibration_method}")
    print(f"   PR-AUC: {calibration_results.loc[calibration_results['Model'] == best_calibration_method, 'PR-AUC'].values[0]:.3f}")
    
    # Визуализация финальной модели
    y_proba_final = final_model.predict_proba(X_test)[:, 1]
    y_pred_final = (y_proba_final > 0.5).astype(int)
    postfix = best_model_name.lower().replace(' ', '_')
    rocPlot(y_test, y_proba_final, f"{postfix}_{type_data}")
    erorrMatrixPlot(y_test, y_proba_final, f"{postfix}_{type_data}")
    
    # Создание ансамблей
    print("\nСОЗДАНИЕ АНСАМБЛЕЙ")
    print("=" * 30)
    
   # Voting ансамбль
    voting_ensemble = create_voting_ensemble([
        ('rf', RandomForestClassifier(**forest_params, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced' if use_smote else None)),
        ('gb', GradientBoostingClassifier(**grad_params, random_state=RANDOM_STATE))
    ])
    
    voting_model, voting_metrics = evaluate_ensemble(
        voting_ensemble, X_train, y_train, X_test, y_test, 'ensemble_forest_boosting', use_smote, sampling_strategy
    )
    
    # Stacking ансамбль
    stacking_ensemble = create_stacking_ensemble([
        ('lr', LogisticRegression(max_iter=100, random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(**forest_params, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced' if use_smote else None))
    ], LogisticRegression(max_iter=100, random_state=RANDOM_STATE))
    
    stacking_model, stacking_metrics = evaluate_ensemble(
        stacking_ensemble, X_train, y_train, X_test, y_test, 'ensemble_forest_logistic', use_smote, sampling_strategy
    )
    
    # Создаем итоговую таблицу с результатами
    final_model_name = best_model_name.lower().replace(' ', '_')
    result_models = pd.DataFrame([
        {
            'model': final_model_name,
            'ROC-AUC': roc_auc_score(y_test, y_proba_final),
            'PR-AUC': average_precision_score(y_test, y_proba_final),
            'Accuracy': accuracy_score(y_test, y_pred_final),
            'F1-Score': f1_score(y_test, y_pred_final),
            'Precision': precision_score(y_test, y_pred_final),
            'Recall': recall_score(y_test, y_pred_final)
        },
        {
            'model': 'ensemble_forest_boosting',
            **voting_metrics
        },
        {
            'model': 'ensemble_forest_logistic',
            **stacking_metrics
        }
    ])
    
    # Сохраняем модели
    save_model(final_model, f"../models/final_model_{final_model_name}_{type_data}.pkl")
    save_model(voting_model, f"../models/ensemble_forest_boosting_{type_data}.pkl")
    save_model(stacking_model, f"../models/ensemble_forest_logistic_{type_data}.pkl")
   
    #  Итоговый отчет
    print(f"""
ИТОГОВЫЕ РЕЗУЛЬТАТЫ:
-----------------------------------------------------------
Лучшая модель: {best_model_name} с {best_calibration_method}
ROC-AUC: {roc_auc_score(y_test, y_proba_final):.3f} 
PR-AUC: {average_precision_score(y_test, y_proba_final):.3f} 
F1-Score: {f1_score(y_test, y_pred_final):.3f}
Precision: {precision_score(y_test, y_pred_final):.3f} (точность предсказаний оттока)
Recall: {recall_score(y_test, y_pred_final):.3f} (находим {recall_score(y_test, y_pred_final):.1%} ушедших)
""")
    
    return {
        'results_df': results_df,
        'final_model': final_model,
        'voting_model': voting_model,
        'stacking_model': stacking_model,
        'result_models': result_models
    }

# =========== ЗАПУСК ПАЙПЛАЙНА ===========
if __name__ == "__main__":
    
    # Можете запускать для разных типов данных
    data_types = [
        ('unbalanced', None),           # Без балансировки
        ('balanced_medium', 0.7),       # 70% minority class (средняя балансировка)
        ('balanced', 1.0),              # 100% (полная балансировка 50/50)
    ]
    
    for type_data, sampling_strategy in data_types:
        try:
            run_pipeline(type_data, sampling_strategy)
            print(f"\nЗавершено для {type_data}")
        except Exception as e:
            print(f"\nОшибка при обработке {type_data}: {e}")