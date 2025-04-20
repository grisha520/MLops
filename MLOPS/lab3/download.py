import pandas as pd
import logging
import sys

def configure_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('car_data_processing.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def download_data():
    """
    Загружает датасет автомобилей и сохраняет его локально.

    Returns:
        pd.DataFrame: Загруженный DataFrame
    """
    try:
        logging.info("Загрузка датасета автомобилей...")
        url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/usedcars.csv'
        df = pd.read_csv(url)
        df.to_csv("used_cars.csv", index=False)
        logging.info(f"Данные успешно загружены и сохранены (всего записей: {len(df)})")
        return df
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        raise

def clean_data(path2df):
    """
    Очищает и обрабатывает данные автомобилей.

    Args:
        path2df (str): Путь к CSV-файлу с данными

    Returns:
        bool: True, если обработка успешна
    """
    try:
        logging.info("Очистка и подготовка данных...")
        df = pd.read_csv(path2df)

        df.dropna(inplace=True)

        df['year'] = df['year'].astype(int)
        df['price'] = df['price'].astype(float)
        df['mileage'] = df['mileage'].astype(float)

        df = df[(df['price'] > 1000) & (df['price'] < 100000)]
        df = df[(df['mileage'] > 0) & (df['mileage'] < 300000)]

        df.to_csv("used_cars_clean.csv", index=False)
        logging.info(f"Очистка завершена: {len(df)} записей сохранено")

        return True
    except Exception as e:
        logging.error(f"Ошибка при обработке данных: {e}")
        raise

if __name__ == "__main__":
    configure_logging()
    try:
        download_data()
        clean_data("used_cars.csv")
        logging.info("Все этапы завершены успешно!")
    except Exception as e:
        logging.critical(f"Критическая ошибка: {e}")
        sys.exit(1)
