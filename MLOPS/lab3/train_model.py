import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib

def scale_data(X, y):
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scaled = scaler.fit_transform(X)
    y_scaled = power_trans.fit_transform(y.values.reshape(-1, 1))
    return X_scaled, y_scaled, scaler, power_trans

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    # Создание директорий
    os.makedirs("models", exist_ok=True)
    
    # Загрузка данных
    df = pd.read_csv("data/processed/cars_processed.csv")
    X = df.drop(columns=['Price(euro)'])
    y = df['Price(euro)']
    
    # Масштабирование данных
    X_scaled, y_scaled, scaler, power_trans = scale_data(X, y)
    
    # Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.3, random_state=42
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
    mlflow.set_experiment("cars_price_prediction")
    
    with mlflow.start_run():
        # Обучение модели
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4)
        clf.fit(X_train, y_train.ravel())
        best_model = clf.best_estimator_
        
        # Предсказание и оценка
        y_pred = best_model.predict(X_val)
        y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1, 1))
        y_val_actual = power_trans.inverse_transform(y_val)
        
        rmse, mae, r2 = eval_metrics(y_val_actual, y_price_pred)
        
        # Логирование параметров и метрик
        mlflow.log_params(best_model.get_params())
        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })
        
        # Логирование модели
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            registered_model_name="cars_price_predictor"
        )
        
        # Сохранение дополнительных артефактов
        joblib.dump(scaler, "models/scaler.joblib")
        joblib.dump(power_trans, "models/power_transformer.joblib")
        
        # Сохранение пути к лучшей модели
        model_uri = mlflow.get_artifact_uri("model")
        with open("models/best_model.txt", "w") as f:
            f.write(model_uri)