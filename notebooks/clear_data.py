import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Пути
RAW_PATH = "../data/raw_data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = "../data/processed"

# Проверяем существование файла
if not os.path.exists(RAW_PATH):
    print(f"Файл не найден: {RAW_PATH}")
    print("Скачайте данные и поместите в data/raw_data/")
    exit(1)


df = pd.read_csv(RAW_PATH)
print(f"   Исходный размер: {df.shape}")

# Убираем tenure = 0 возможно это новые клиенты, а они нам ничего не дадут в выборке
df = df[df['tenure'] != 0]

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

nan_counts = df.isna().sum()
print("   NaN по столбцам:")
for col, count in nan_counts[nan_counts > 0].items():
    print(f"     {col}: {count} NaN")

# Проверяем и удаляем дубликаты
duplicates = df.duplicated().sum()
print(f"   Найдено дубликатов: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"   Дубликаты удалены")

# Удаляем customerID (неинформативный признак)
df = df.drop('customerID', axis=1)


# Добавление новых признаков
# ЦЕННОСТЬ КЛИЕНТА (Customer Lifetime Value) Средний доход в месяц
df['CLV'] = df['TotalCharges'] / df['tenure']

# РИСКОВЫЙ СКОР 
df['RiskScore'] = df['MonthlyCharges'] / df['tenure']

# ПЛОТНОСТЬ УСЛУГ
services = ['PhoneService', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies']
df['TotalServices'] = df[services].apply(lambda x: (x == 'Yes').sum(), axis=1)
df['ServiceDensity'] = df['TotalServices'] / df['tenure']


# Кодируем категориальные переменные
print("\n Кодирование категориальных переменных")

# Бинарные колонки (Yes/No)
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
        print(f"   {col}: Yes→1, No→0")

# Пол (Male/Female)
if 'gender' in df.columns:
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    print(f"   gender: Male→1, Female→0")


# One-Hot Encoding для остальных категориальных
cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

# Разделение на признаки и целевую переменную
X = df.drop('Churn', axis=1)
y = df['Churn']

# Разделение на тренировочную и тестовую выборки
# Разделение делается чтобы сбалансировать обучающую выборку и не трогать тестовые данные
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,stratify=y,random_state=42)

print("\nРаспределение классов:")
print(f"Train: класс 0 - {(y_train == 0).sum()}, класс 1 - {(y_train == 1).sum()}")
print(f"Test:  класс 0 - {(y_test == 0).sum()}, класс 1 - {(y_test == 1).sum()}")

ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
ohe.fit(X_train[cat_cols]) 

X_train_encoded = ohe.transform(X_train[cat_cols])
X_test_encoded = ohe.transform(X_test[cat_cols])

feature_names = ohe.get_feature_names_out(cat_cols)
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=feature_names, index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=feature_names, index=X_test.index)

# Удаляем старые категориальные колонки и объединяем с закодированными
X_train = X_train.drop(cat_cols, axis=1)
X_test = X_test.drop(cat_cols, axis=1)

X_train = pd.concat([X_train, X_train_encoded_df], axis=1)
X_test = pd.concat([X_test, X_test_encoded_df], axis=1)


# Сохраняем encoder для тестовых данных
os.makedirs('encoders', exist_ok=True)
joblib.dump(ohe, 'encoders/onehot_encoder.pkl')


# Нормализация числовых признаков
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'CLV', 'RiskScore', 'TotalServices', 'ServiceDensity']
scaler = StandardScaler()
# df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])


# Сохраняем обработанные данные
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Сохраняем несбалансированные тренировочные данные
pd.concat([X_train, y_train], axis=1).to_csv(f'{PROCESSED_DIR}/train_unbalanced.csv', index=False)

# Сохраняем тестовые данные (НЕ балансируем!)
pd.concat([X_test, y_test], axis=1).to_csv(f'{PROCESSED_DIR}/test.csv', index=False)

# Сохраняем полные данные
pd.concat([X, y], axis=1).to_csv(f'{PROCESSED_DIR}/full_processed.csv', index=False)