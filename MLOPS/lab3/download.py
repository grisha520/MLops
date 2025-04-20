import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import logging
import sys

def configure_logging():
    """Настройка логирования для отслеживания выполнения скрипта"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_processing.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def download_data():
    """
    Загружает данные из удаленного CSV-файла и сохраняет их локально
    
    Returns:
        pd.DataFrame: Загруженный DataFrame с исходными данными
    """
    try:
        logging.info("Начало загрузки данных...")
        url = 'https://raw.githubusercontent.com/dayekb/Basic_ML_Alg/main/cars_moldova_no_dup.csv'
        df = pd.read_csv(url, delimiter=',')
        df.to_csv("cars.csv", index=False)
        logging.info(f"Данные успешно загружены. Сохранено {len(df)} записей в cars.csv")
        return df
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {str(e)}")
        raise

def clean_data(path2df):
    """
    Очищает и преобразует данные автомобилей
    
    Args:
        path2df (str): Путь к CSV-файлу с исходными данными
    
    Returns:
        bool: True если обработка прошла успешно
    """
    try:
        logging.info("Начало обработки данных...")
        df = pd.read_csv(path2df)
        
        # Определение столбцов
        cat_columns = ['Make', 'Model', 'Style', 'Fuel_type', 'Transmission']
        num_columns = ['Year', 'Distance', 'Engine_capacity(cm3)', 'Price(euro)']
        
        # Применение правил очистки данных
        cleaning_rules = [
            ((df.Year < 2021) & (df.Distance < 1100), "Удаление машин с подозрительно низким пробегом для их возраста"),
            (df.Distance > 1e6, "Удаление машин с чрезмерно высоким пробегом (>1,000,000 км)"),
            (df["Engine_capacity(cm3)"] < 200, "Удаление машин с объемом двигателя <200 см³"),
            (df["Engine_capacity(cm3)"] > 5000, "Удаление машин с объемом двигателя >5000 см³"),
            (df["Price(euro)"] < 101, "Удаление машин с ценой <101 евро"),
            (df["Price(euro)"] > 1e5, "Удаление машин с ценой >100,000 евро"),
            (df.Year < 1971, "Удаление машин старше 1971 года")
        ]
        
        initial_count = len(df)
        for condition, description in cleaning_rules:
            rows_to_drop = df[condition]
            if not rows_to_drop.empty:
                logging.info(f"{description}: удалено {len(rows_to_drop)} записей")
                df = df.drop(rows_to_drop.index)
        
        # Кодирование категориальных переменных
        logging.info("Кодирование категориальных переменных...")
        encoder = OrdinalEncoder()
        df[cat_columns] = encoder.fit_transform(df[cat_columns])
        
        # Сохранение обработанных данных
        df.to_csv('df_clean.csv', index=False)
        final_count = len(df)
        logging.info(f"Обработка завершена. Исходно: {initial_count} записей, после очистки: {final_count} записей")
        
        return True
    except Exception as e:
        logging.error(f"Ошибка при обработке данных: {str(e)}")
        raise

if __name__ == "__main__":
    configure_logging()
    try:
        # Загрузка и обработка данных
        download_data()
        clean_data("cars.csv")
        logging.info("Все этапы обработки данных успешно завершены!")
    except Exception as e:
        logging.critical(f"Критическая ошибка в процессе выполнения: {str(e)}")
        sys.exit(1)
