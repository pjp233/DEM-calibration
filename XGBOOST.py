# 导入必要的库
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt

# 1. 读取数据
data = pd.read_excel('E:\\python3.8-study\\shuju3.0.xlsx')  # 读取数据文件

# 2. 提取输入和输出特征
X = data.iloc[:, 4:9].values  # 输入特征
Y = data.iloc[:, :4].values   # 输出目标

# 3. 数据归一化处理
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_normalized = scaler_X.fit_transform(X)
Y_normalized = scaler_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y_normalized, test_size=0.3, random_state=42)

# 4. 定义参数搜索空间
param_grid = {
    'n_estimators': [50, 100, 150, 200],  # 树的数量
    'max_depth': [3, 6, 9, 12],           # 最大深度
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # 学习率
    'subsample': [0.8, 1.0],              # 采样比例
    'colsample_bytree': [0.8, 1.0],       # 每棵树的特征选择比例
    'gamma': [0, 0.1, 0.2],               # 最小损失减少
    'reg_alpha': [0, 0.1, 1],             # L1 正则化
    'reg_lambda': [1, 2, 5],              # L2 正则化
}

# 5. 使用 GridSearchCV 进行超参数搜索
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    verbose=2
)

# 6. 模型训练
grid_search.fit(X_train, Y_train)

# 输出最佳超参数
print("最佳超参数组合：", grid_search.best_params_)

# 7. 使用最佳模型进行预测
best_model = grid_search.best_estimator_

# 8. 进行预测
predictions_normalized = best_model.predict(X_test)

# 9. 反归一化预测结果
predictions_normalized = predictions_normalized.reshape(-1, 4)
predictions = scaler_Y.inverse_transform(predictions_normalized)

# 10. 输出预测结果
print("预测结果：")
print(predictions)

Y_test_re = scaler_Y.inverse_transform(Y_test)  # 真实值
print("真实值：")
print(Y_test_re)

# 11. 计算误差指标
rmse = np.sqrt(mean_squared_error(Y_test_re, predictions))
mse = mean_squared_error(Y_test_re, predictions)
mae = mean_absolute_error(Y_test_re, predictions)
mape = np.mean(np.abs((Y_test_re - predictions) / Y_test_re)) * 100
smape = 100 / len(Y_test_re) * np.sum(2 * np.abs(Y_test_re - predictions) / (np.abs(Y_test_re) + np.abs(predictions)))
R2 = r2_score(Y_test_re, predictions)

# 输出评估指标
print(f"RMSE: {rmse:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R2: {R2:.4f}")
print(f"SMAPE: {smape:.2f}%")

# 12. 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(predictions[:, 0], label='预测值 (输出1)')
plt.plot(Y_test_re[:, 0], label='真实值 (输出1)', linestyle='dashed')
plt.xlabel('样本索引')
plt.ylabel('预测值')
plt.title('XGBoost 回归预测结果')
plt.legend()
plt.show()

# 13. 保存预测结果
save_path = "E:\\python3.8-study\\figure\\"
pred_file_path = save_path + "XGB_test_pre.xlsx"
true_file_path = save_path + "XGB_test_true.xlsx"

pd.DataFrame(predictions, columns=['杨氏模量_预测', '粘聚力_预测', '摩擦角_预测', '抗拉强度_预测']).to_excel(pred_file_path, index=False)
pd.DataFrame(Y_test_re, columns=['杨氏模量_真实', '粘聚力_真实', '摩擦角_真实', '抗拉强度_真实']).to_excel(true_file_path, index=False)
