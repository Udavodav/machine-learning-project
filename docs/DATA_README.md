# Описание используемых данных

## Обзор набора данных

**Название:** Telco Customer Churn Dataset  
**Источник:** Kaggle - Telco Customer Churn Dataset  
**Дата загрузки:** 01.12.25  
**Цель проекта:** Прогнозирование оттока клиентов телеком-компании (бинарная классификация)

## Описание данных

Набор данных содержит информацию о 7043 клиентах телеком-компании за последний квартал. Каждая строка представляет одного клиента, каждый столбец содержит атрибуты клиента, описанные в словаре данных ниже.

## Словарь данных (Data Dictionary)
| Название столбца     | Тип данных  | Описание                                             | Пример                                                     |
|----------------------|-------------|------------------------------------------------------|------------------------------------------------------------|
| **customerID**       | string      | Уникальный идентификатор клиента                     | 7590-VHVEG                                                 |
| **gender**           | categorical | Пол клиента                                          | Male, Female                                               |
| **SeniorCitizen**    | binary      | Является ли клиент пенсионером (1, 0)                | 0, 1                                                       |
| **Partner**          | categorical | Есть ли у клиента партнер                            | Yes, No                                                    |
| **Dependents**       | categorical | Есть ли у клиента иждивенцы                          | Yes, No                                                    |
| **tenure**           | integer     | Количество месяцев, которые клиент провел в компании | 1, 72                                                      |
| **PhoneService**     | categorical | Есть ли у клиента телефонная служба                  | Yes, No                                                    |
| **MultipleLines**    | categorical | Есть ли у клиента несколько линий                    | Yes, No, No phone service                                  |
| **InternetService**  | categorical | Тип интернет-сервиса клиента                         | DSL, Fiber optic, No                                       |
| **OnlineSecurity**   | categorical | Есть ли у клиента онлайн-безопасность                | Yes, No, No internet service                               |
| **OnlineBackup**     | categorical | Есть ли у клиента онлайн-резервное копирование       | Yes, No, No internet service                               |
| **DeviceProtection** | categorical | Есть ли у клиента защита устройства                  | Yes, No, No internet service                               |
| **TechSupport**      | categorical | Есть ли у клиента техническая поддержка              | Yes, No, No internet service                               |
| **StreamingTV**      | categorical | Есть ли у клиента стриминг ТВ                        | Yes, No, No internet service                               |
| **StreamingMovies**  | categorical | Есть ли у клиента стриминг фильмов                   | Yes, No, No internet service                               |
| **Contract**         | categorical | Тип контракта клиента                                | Month-to-month, One year, Two year                         |
| **PaperlessBilling** | categorical | Использует ли клиент безбумажный биллинг             | Yes, No                                                    |
| **PaymentMethod**    | categorical | Способ оплаты                                        | Electronic check, Mailed check, Bank transfer, Credit card |
| **MonthlyCharges**   | float       | Ежемесячные платежи                                  | 29.85, 56.95                                               |
| **TotalCharges**     | string      | Общая сумма платежей                                 | 29.85, 1889.5                                              |
| **Churn**            | categorical | Целевая переменная: Ушел ли клиент                   | Yes, No                                                    |


## Структура данных

**Тип:** Табличные данные (CSV)  
**Размер:** 7043 строк × 21 столбец  
**Объем:** ~500 КБ  
**Формат файла**: .csv  
**Кодировка:** UTF-8  
**Разделитель:** Запятая

## Типы данных

**Количественные признаки (numerical):** 3
**Категориальные признаки (categorical):** 17
**Бинарные признаки:** 1 (SeniorCitizen)


## Ограничения и предостережения

TotalCharges: Некоторые значения хранятся как строки с пробелами  
Кодировка пропусков: "No internet service" и "No phone service" по сути являются пропусками для соответствующих сервисов  
Мультиколлинеарность (зависимость): Есть зависимость между услугами (например, если InternetService = "No", то все онлайн-услуги недоступны)  

## Особенности обработки

Требуемая предобработка:

TotalCharges из string в float  
Пустые TotalCharges (вероятно, новые клиенты) заполнить 0 или удалить  
One-Hot Encoding для номинальных признаков  
Ordinal Encoding для порядковых (Contract)

## Схемы зависимостей
Логические зависимости между признаками:

InternetService = "No" →  
    OnlineSecurity = "No internet service"  
    OnlineBackup = "No internet service"  
    DeviceProtection = "No internet service"  
    TechSupport = "No internet service"  
    StreamingTV = "No internet service"  
    StreamingMovies = "No internet service"  

Бизнес-зависимости:

tenure ↑ → TotalCharges ↑
Contract = "Two year" → Churn ↓ (ожидаемо)
MonthlyCharges ↑ → Churn ↑ (ожидаемо)



