import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    df = pd.read_csv(
        'https://raw.githubusercontent.com/dayekb/Basic_ML_Alg/main/cars_moldova_no_dup.csv', 
        delimiter=','
    )
    df.to_csv("data/raw/cars.csv", index=False)
    return df

def clear_data(path2df):
    df = pd.read_csv(path2df)
    
    cat_columns = ['Make', 'Model', 'Style', 'Fuel_type', 'Transmission']
    num_columns = ['Year', 'Distance', 'Engine_capacity(cm3)', 'Price(euro)']
    
    # Очистка данных
    question_dist = df[(df.Year < 2021) & (df.Distance < 1100)]
    df = df.drop(question_dist.index)
    
    question_dist = df[(df.Distance > 1e6)]
    df = df.drop(question_dist.index)
    
    question_engine = df[df["Engine_capacity(cm3)"] < 200]
    df = df.drop(question_engine.index)
    
    question_engine = df[df["Engine_capacity(cm3)"] > 5000]
    df = df.drop(question_engine.index)
    
    question_price = df[(df["Price(euro)"] < 101)]
    df = df.drop(question_price.index)
    
    question_price = df[df["Price(euro)"] > 1e5]
    df = df.drop(question_price.index)
    
    question_year = df[df.Year < 1971]
    df = df.drop(question_year.index)
    
    df = df.reset_index(drop=True)
    
    # Кодирование категориальных признаков
    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])
    
    # Сохранение обработанных данных
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv('data/processed/cars_processed.csv', index=False)
    
    # Сохранение кодировщика
    joblib.dump(ordinal, 'models/ordinal_encoder.joblib')
    
    return True

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    download_data()
    clear_data("data/raw/cars.csv")