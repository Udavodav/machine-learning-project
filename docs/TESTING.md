# Стратегия тестирования и покрытие кода

## Обзор тестирования

Проект использует комплексную стратегию тестирования, включающую различные уровни тестов для обеспечения надежности и качества модели машинного обучения.

## Структура тестов

```text
tests/
├── baseline_test.py       # Базовые проверки модели
├── e2e_test.py           # End-to-end тестирование
├── integration_test.py    # Интеграционные тесты
├── quality_of_data_test.py # Тесты качества данных
├── reproduct_test.py     # Тесты воспроизводимости
└── unit_test.py          # Юнит-тесты
```


## Типы тестов

### 1. Базовые тесты (Baseline Tests)

**Цель:** Проверка минимальных требований к модели

- `test_baseline_comparison` - Сравнение с базовой моделью
- `test_minimum_roc_auc_threshold` - Проверка ROC-AUC ≥ 0.75
- `test_overfitting_check` - Проверка на переобучение
- `test_model_loading` - Проверка загрузки модели

### 2. End-to-End тесты

**Цель:** Проверка полного цикла работы

- `test_end_to_end_pipeline_minimal` - Минимальный E2E пайплайн

### 3. Интеграционные тесты

**Цель:** Проверка взаимодействия компонентов

- `test_calibrate_model_integration` - Калибровка модели
- `test_ensemble_creation_functions` - Создание ансамблей
- `test_model_saving_and_loading` - Сохранение и загрузка

### 4. Тесты качества данных

**Цель:** Обеспечение качества входных данных

- `test_no_target_leakage` - Отсутствие утечки целевой переменной
- `test_missing_values_after_processing` - Обработка пропусков
- `test_class_balance_report` - Баланс классов

### 5. Тесты воспроизводимости

**Цель:** Гарантия воспроизводимости результатов

- `test_reproducibility_with_fixed_seed` - Воспроизводимость с фиксированным seed
- `test_deterministic_pipeline` - Детерминированность пайплайна
- `test_cross_validation_reproducibility` - Воспроизводимость CV

### 6. Юнит-тесты

**Цель:** Проверка отдельных функций

- `test_randForestOpt_function` - Оптимизация Random Forest
- `test_gradOpt_function` - Оптимизация Gradient Boosting
- `test_calculate_metrics_function` - Расчет метрик
- `test_print_metrics_function` - Вывод метрик
- `test_load_data_function` - Загрузка данных
- `test_split_data_for_calibration` - Разделение для калибровки

## Стратегия тестирования

### Принципы

1. **Пирамида тестирования:** Больше юнит-тестов, меньше E2E тестов
2. **Изоляция:** Тесты независимы друг от друга
3. **Детерминизм:** Результаты воспроизводимы
4. **Скорость:** Быстрое выполнение тестовой базы


## Запуск тестов

```bash
# Все тесты
pytest tests/

# С подробным выводом
pytest tests/ -v

# Конкретный тестовый файл
pytest tests/unit_test.py

# Конкретный тест
pytest tests/unit_test.py::test_randForestOpt_function
```
