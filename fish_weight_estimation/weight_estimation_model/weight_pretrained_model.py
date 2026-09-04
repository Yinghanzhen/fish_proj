import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib
# 设置绘图字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据 (Excel)
excel_path = 'fish_body_data.xlsx'
df = pd.read_excel(excel_path)

print(f"\n总样本数: {len(df)}")

# 定义特征变量 X 和目标变量 y
X = df[['体长', '体高']]  # 特征
y = df['体重']  # 目标

# 2. 划分训练集与测试集 (8:2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 对于神经网络，需要进行特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 辅助评估函数与结果收集器
results = []
model_names = []  #用于记录模型名称

def evaluate_and_record(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    res = {
        'Model': model_name,
        'R2': r2,
        'RMSE (g)': rmse,
        'MAE (g)': mae,
        'MAPE (%)': mape
    }
    results.append(res)

    print(f"\n{model_name} 模型性能:")
    print(f" - R² (决定系数): {r2:.4f}")
    print(f" - RMSE (均方根误差): {rmse:.2f} g")
    print(f" - MAE (平均绝对误差): {mae:.2f} g")
    print(f" - MAPE (平均绝对百分比误差): {mape:.2f}%")
    return y_pred

# 存储预测结果用于绘图
predictions = {}
models_to_save = {}  # 用于保存模型对象

# 4. 训练各个模型
# (1) 多元线性回归
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
models_to_save['linear_regression'] = lr_model
predictions['Linear Regression'] = evaluate_and_record(
    y_test, lr_model.predict(X_test), 'Linear Regression'
)

# (2) 岭回归
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train, y_train)
models_to_save['ridge'] = ridge_model
predictions['Ridge Regression'] = evaluate_and_record(
    y_test, ridge_model.predict(X_test), 'Ridge Regression'
)

# (3) 随机森林回归
rf_model = RandomForestRegressor(
    n_estimators=100, max_depth=6, random_state=42
)
rf_model.fit(X_train, y_train)
models_to_save['random_forest'] = rf_model
predictions['Random Forest'] = evaluate_and_record(
    y_test, rf_model.predict(X_test), 'Random Forest'
)

# (4) XGBoost 回归
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
models_to_save['xgboost'] = xgb_model
predictions['XGBoost'] = evaluate_and_record(
    y_test, xgb_model.predict(X_test), 'XGBoost'
)

# (5) LightGBM 回归
lgb_model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=15,
    random_state=42,
    verbosity=-1
)
lgb_model.fit(X_train, y_train)
models_to_save['lightgbm'] = lgb_model
predictions['LightGBM'] = evaluate_and_record(
    y_test, lgb_model.predict(X_test), 'LightGBM'
)

# (6) 五层神经网络 (新增)
# 使用 scaled 数据进行训练
mlp_model = MLPRegressor(
    hidden_layer_sizes=(256,128,64, 32, 16),  # 三个隐藏层，分别为64、32、16个神经元
    activation='relu',                # 激活函数
    solver='adam',                   # 优化器
    alpha=0.001,                     # L2正则化系数
    batch_size='auto',
    learning_rate='adaptive',        # 自适应学习率
    learning_rate_init=0.001,
    max_iter=1000,                   # 最大迭代次数
    early_stopping=True,             # 使用早停防止过拟合
    validation_fraction=0.1,
    n_iter_no_change=20,             # 早停耐心值
    random_state=42,
    verbose=False
)
mlp_model.fit(X_train_scaled, y_train)
models_to_save['neural_network'] = mlp_model
models_to_save['scaler'] = scaler  # 保存标准化器，用于后续预测

# 预测时需要先标准化
y_pred_mlp = mlp_model.predict(X_test_scaled)
predictions['Neural Network'] = evaluate_and_record(
    y_test, y_pred_mlp, 'Neural Network (5层)'
)

# 5. 生成论文对比表格
df_summary = pd.DataFrame(results)
print(df_summary.to_string(index=False))

# 6. 可视化对比图 (3x2 散点对比图)
fig, axes = plt.subplots(3, 2, figsize=(14, 15))
axes = axes.flatten()

model_keys = [
    ('Linear Regression', '多元线性回归', 'blue'),
    ('Ridge Regression', '岭回归', 'cyan'),
    ('Random Forest', '随机森林', 'green'),
    ('XGBoost', 'XGBoost', 'darkorange'),
    ('LightGBM', 'LightGBM', 'purple'),
    ('Neural Network', '多层神经网络', 'crimson')
]

for idx, (key, label_cn, color) in enumerate(model_keys):
    ax = axes[idx]
    y_pred = predictions[key]
    r2_val = df_summary.loc[df_summary['Model'].str.contains(key, case=False, na=False), 'R2'].values[0]

    ax.scatter(y_test, y_pred, color=color, alpha=0.7, edgecolors='k', label='测试集样本')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='1:1 理想参考线')

    ax.set_title(f"{label_cn} (R² = {r2_val:.4f})", fontsize=12, fontweight='bold')
    ax.set_xlabel("真实体重 (g)", fontsize=10)
    ax.set_ylabel("预测体重 (g)", fontsize=10)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# 7. 保存模型

# 保存所有模型
joblib.dump(lr_model, 'model_linear_regression.pkl')
joblib.dump(ridge_model, 'model_ridge.pkl')
joblib.dump(rf_model, 'model_random_forest.pkl')
joblib.dump(xgb_model, 'model_xgboost.pkl')
joblib.dump(lgb_model, 'model_lightgbm.pkl')
joblib.dump(mlp_model, 'model_neural_network.pkl')
joblib.dump(scaler, 'scaler.pkl')  # 保存标准化器

# 保存配置
config = {
    'k_factor': 0.962,
    'features': ['体长', '体高'],
    'neural_network_architecture': '2-64-32-16-1'
}
joblib.dump(config, 'system_config.pkl')

def estimate_fish_weight(top_fish_list, side_fish_list):
    """提取特征，加载多模型计算平均体重并写回顶视角与侧视角鱼类实例中"""
    if len(top_fish_list) != len(side_fish_list):
        raise ValueError(
            f"输入列表长度不一致: 俯视角 ({len(top_fish_list)}) vs 侧视角 ({len(side_fish_list)})"
        )
    if not top_fish_list:
        return

    # 1. 提取特征矩阵 [体长, 体高]
    features = [
        [top.spine_length, side.body_height_3d]
        for top, side in zip(top_fish_list, side_fish_list)
    ]
    X = np.array(features, dtype=np.float64)

    # 2. 定义模型配置
    models_config = [
        ("多元线性回归", "model_linear_regression.pkl", False),
        ("岭回归", "model_ridge.pkl", False),
        ("随机森林", "model_random_forest.pkl", False),
        ("XGBoost", "model_xgboost.pkl", False),
        ("LightGBM", "model_lightgbm.pkl", False),
        ("神经网络", "model_neural_network.pkl", True),
    ]

    # 加载标准化特征用于神经网络
    scaler = joblib.load("scaler.pkl")
    X_scaled = scaler.transform(X)

    # 3. 批量推理收集预测结果
    preds_list = []
    for name, pkl_path, need_scale in models_config:
        model = joblib.load(pkl_path)
        input_data = X_scaled if need_scale else X
        pred = model.predict(input_data)
        preds_list.append(pred)

    # 4. 计算所有模型输出的平均体重 (Shape: [N_fishes])
    # preds_array Shape 为 [N_models, N_fishes] -> axis=0 求均值得到 [N_fishes]
    avg_weights = np.mean(np.array(preds_list), axis=0)

    # 5. 将平均体重赋予顶视角和侧视角的实例中
    for top, side, weight in zip(top_fish_list, side_fish_list, avg_weights):
        weight_val = float(weight)
        top.weight = weight_val
        side.weight = weight_val