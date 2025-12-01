import os
import sys

# Настройка пути для сохранения данных
DATA_DIR = "data/raw_data"
DATASET_NAME = "drnimishadavis/telco-customer-churn-dataset"
OUTPUT_FILE = "Telco-Customer-Churn.csv"

def install_kaggle():
    try:
        import kaggle
        print("Kaggle API уже установлен")
        return True
    except ImportError:
        print("Устанавливаю Kaggle API...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
            print("Kaggle API установлен")
            return True
        except Exception as e:
            print(f"Ошибка установки: {e}")
            return False

def check_kaggle_auth():
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_file = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(kaggle_file):
        print("Файл с учетными данными найден")
        return True
    else:
        print("Файл kaggle.json не найден!")
        print("\n📝 Инструкция по настройке:")
        print("1. Зарегистрируйтесь на Kaggle: https://www.kaggle.com")
        print("2. Перейдите в настройки аккаунта")
        print("3. Нажмите 'Create New API Token'")
        print("4. Скачанный файл kaggle.json поместите в папку ~/.kaggle/")
        return False

def download_dataset():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print(f"Скачиваю датасет: {DATASET_NAME}")
        
        # Создаем директорию для данных
        os.makedirs(DATA_DIR, exist_ok=True)
        
        api = KaggleApi()
        api.authenticate()
        
        api.dataset_download_files(
            DATASET_NAME,
            path=DATA_DIR,
            unzip=True,
            quiet=False
        )
        
        downloaded_files = os.listdir(DATA_DIR)
        for file in downloaded_files:
            if file.endswith(".csv"):
                print(f"Файл скачан: {DATA_DIR}/{file}")
                return True
        
        print("CSV файлы не найдены в скачанном архиве")
        return False
        
    except Exception as e:
        print(f"Ошибка при скачивании: {e}")
        return False


def main():
    print("=" * 50)
    print("Скачивание данных с Kaggle")
    print("=" * 50)
    
    # Проверяем, не скачаны ли данные уже
    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")] if os.path.exists(DATA_DIR) else []
    if data_files:
        print(f"Данные уже скачаны в папке {DATA_DIR}/")
        print(f"   Найдены файлы: {', '.join(data_files)}")
        return True
    
    # Устанавливаем Kaggle API
    if not install_kaggle():
        return False
    
    # Проверяем учетные данные
    if not check_kaggle_auth():
        return False
    
    # Скачиваем датасет
    if download_dataset():
        print("\nДанные успешно скачаны!")
        return True
    else:
        print("\nСкачивание не удалось")
        print("Проверьте:")
        print("1. Установлен ли Kaggle API: pip install kaggle")
        print("2. Настроены ли учетные данные: файл ~/.kaggle/kaggle.json")
        print("3. Есть ли доступ к датасету: https://www.kaggle.com/datasets/drnimishadavis/telco-customer-churn-dataset")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)