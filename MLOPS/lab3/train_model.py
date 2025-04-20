import os
from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='train_model.log'
)

def scale_frame(frame):
    """Масштабирование данных"""
    df = frame.copy()
    X, y = df.drop(columns=['Price(euro)']), df['Price(euro)']
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(X.values)
    y_scale = power_trans.fit_transform(y.values.reshape(-1,1))
    return X_scale, y_scale, power_trans, scaler

def eval_metrics(actual, pred):
    """Вычисление метрик качества модели"""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def save_best_model_path(path):
    """Сохранение пути к лучшей модели в файл"""
    with open("best_model.txt", "w") as f:
        f.write(path)
    logging.info(f"Saved best model path to best_model.txt: {path}")

if __name__ == "__main__":
    try:
        logging.info("Starting model training process...")
        
        # Загрузка данных
        df = pd.read_csv("./df_clean.csv")
        logging.info(f"Loaded data with shape: {df.shape}")
        
        # Масштабирование данных
        X, y, power_trans, scaler = scale_frame(df)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Параметры для GridSearch
        params = {
            'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
            'l1_ratio': [0.001, 0.05, 0.01, 0.2],
            "penalty": ["l1", "l2", "elasticnet"],
            "loss": ['squared_error', 'huber', 'epsilon_insensitive'],
            "fit_intercept": [False, True],
        }
        
        # Настройка MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("linear_model_cars")
        
        with mlflow.start_run() as run:
            logging.info("MLflow run started")
            
            # Обучение модели
            lr = SGDRegressor(random_state=42)
            clf = GridSearchCV(lr, params, cv=3, n_jobs=4)
            clf.fit(X_train, y_train.reshape(-1))
            
            # Получение лучшей модели
            best = clf.best_estimator_
            logging.info(f"Best model parameters: {best.get_params()}")
            
            # Предсказание и оценка
            y_pred = best.predict(X_val)
            y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1,1))
            rmse, mae, r2 = eval_metrics(
                power_trans.inverse_transform(y_val), 
                y_price_pred
            )
            
            # Логирование параметров
            mlflow.log_params(best.get_params())
            
            # Логирование метрик
            mlflow.log_metrics({
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            })
            
            # Логирование модели
            signature = infer_signature(X_train, best.predict(X_train))
            mlflow.sklearn.log_model(best, "model", signature=signature)
            
            # Сохранение модели
            model_path = "lr_cars_model.pkl"
            joblib.dump({
                'model': best,
                'scaler': scaler,
                'power_trans': power_trans
            }, model_path)
            mlflow.log_artifact(model_path)
            
            logging.info(f"Model training completed with R2: {r2:.4f}")
        
        # Получение пути к лучшей модели
        df_runs = mlflow.search_runs()
        best_run = df_runs.sort_values("metrics.r2", ascending=False).iloc[0]
        best_model_path = best_run['artifact_uri'].replace("file://", "") + "/model"
        
        # Сохранение пути к лучшей модели
        save_best_model_path(best_model_path)
        
        logging.info("Training process completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise
