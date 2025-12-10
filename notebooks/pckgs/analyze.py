import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score, 
                           accuracy_score, confusion_matrix, roc_curve, average_precision_score)


def rocPlot(y, y_proba, postfix_name = "", isSave = True):
    """функция для для построения ROC-кривой"""    
    # Вычисляем ROC-кривую
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    roc_auc = roc_auc_score(y, y_proba)
    
    # Строим график
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайный классификатор (AUC = 0.500)')
    
    # Находим оптимальный порог
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    
    # Отмечаем оптимальную точку
    plt.plot(optimal_fpr, optimal_tpr, 'ro', markersize=10, label=f'Оптимальный порог ({optimal_threshold:.2f})')
    
    # Настраиваем график
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC-кривая модели')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Сохраняем
    if isSave:
        plt.savefig(f"../plots/roc_curve_{postfix_name}.png", dpi=300, bbox_inches='tight')
    plt.show()


def erorrMatrixPlot(y, y_proba, postfix_name = "", isSave = True):

    # Предсказания на оптимальном пороге
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_optimal = (y_proba > optimal_threshold).astype(int)
    
    # Создаём матрицу ошибок
    cm = confusion_matrix(y, y_pred_optimal)
    
    # Преобразуем в DataFrame для красивого отображения
    cm_df = pd.DataFrame(cm, index=['Факт: Не уходит', 'Факт: Уходит'], columns=['Прогноз: Не уходит', 'Прогноз: Уходит'])
    
    # Строим тепловую карту
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.title(f'Матрица ошибок (порог = {optimal_threshold:.2f})')
    plt.tight_layout()
    
    # Сохраняем
    if isSave:
        plt.savefig(f"../plots/error_matrix_{postfix_name}.png", dpi=300, bbox_inches='tight')
    plt.show()


