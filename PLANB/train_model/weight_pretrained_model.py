import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


files = {
    "4_features": ("data_4.xlsx", ["体宽", "体长", "弧度", "背高"]),
    "3_features": ("data_3.xlsx", ["体长", "弧度", "背高"])
}

target = "体重"
os.makedirs("models", exist_ok=True)


# ================= 模型 =================
models = {
    "LinearRegression": LinearRegression(),

    "Ridge": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ),

    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "MLP": make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=3000,
            early_stopping=True,
            random_state=42
        )
    )
}


all_results = []

for dataset_name, (file, features) in files.items():

    df = pd.read_excel(file)
    df = df[features + [target]].dropna()

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs(f"weights/{dataset_name}", exist_ok=True)

    for name, model in models.items():

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mape = np.mean(np.abs((y_test - pred) / y_test)) * 100

        print(
            dataset_name,
            name,
            f"R²={r2:.4f}",
            f"MAE={mae:.2f}",
            f"RMSE={rmse:.2f}",
            f"MAPE={mape:.2f}%"
        )

        # 保存模型
        joblib.dump(
            {
                "model": model,
                "features": features
            },
            f"models/{dataset_name}/{name}.pkl"
        )

        all_results.append([
            dataset_name, name, r2, mae, rmse, mape
        ])


print("\n训练完成！")