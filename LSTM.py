# 导入必要库
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from scikeras.wrappers import KerasRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 数据预处理
data = pd.read_excel('E:\\python3.8-study\\shuju3.0.xlsx')
X = data.iloc[:, 4:9].values  # 输入特征
Y = data.iloc[:, :4].values   # 输出目标

# 清理列名，移除非 ASCII 字符
data.columns = [col.encode('ascii', 'ignore').decode('ascii') for col in data.columns]

# 数据归一化
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_normalized = scaler_X.fit_transform(X)
Y_normalized = scaler_Y.fit_transform(Y)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y_normalized, test_size=0.3, random_state=42)

# 调整 LSTM 输入形状为 3D
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))  # 确保测试集形状一致

def reshape_for_lstm(data):
    return data.reshape((data.shape[0], 1, data.shape[1]))

# 定义模型构建函数
def create_model(learning_rate=0.001, lstm_units=64, dense_units=32):
    model = Sequential([
        LSTM(lstm_units, activation='tanh', input_shape=(1, 5)),  # 确保输入形状匹配
        Dense(dense_units, activation='relu'),
        Dense(4, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# 包装 Keras 模型
model = KerasRegressor(model=create_model, learning_rate=0.001, lstm_units=64, dense_units=32, verbose=0)

# 定义超参数搜索范围
param_grid = {
    'model__learning_rate': [0.0001, 0.001, 0.01],
    'model__lstm_units': [16, 32, 64],
    'model__dense_units': [16, 32, 64],
    'batch_size': [16, 32, 64, 128],
    'epochs': [100, 200, 300]
}

# 执行网格搜索
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_result = grid.fit(X_train, Y_train)

# 输出最佳超参数
print(f"最佳参数: {grid_result.best_params_}")

# 训练集预测与反归一化
predictions_train_normalized = grid_result.best_estimator_.predict(X_train)
predictions_train = scaler_Y.inverse_transform(predictions_train_normalized.reshape(-1, 4))  # 确保形状匹配
Y_train_true = scaler_Y.inverse_transform(Y_train)

# 测试集预测与反归一化
predictions_test_normalized = grid_result.best_estimator_.predict(X_test)
predictions_test = scaler_Y.inverse_transform(predictions_test_normalized.reshape(-1, 4))  # 确保形状匹配
Y_test_true = scaler_Y.inverse_transform(Y_test)

# 计算误差指标
rmse = np.sqrt(mean_squared_error(Y_test_true, predictions_test))
mse = mean_squared_error(Y_test_true, predictions_test)
mae = mean_absolute_error(Y_test_true, predictions_test)
mape = np.mean(np.abs((Y_test_true - predictions_test) / Y_test_true)) * 100
smape = 100 / len(Y_test_true) * np.sum(2 * np.abs(Y_test_true - predictions_test) / (np.abs(Y_test_true) + np.abs(predictions_test)))
R2 = r2_score(Y_test_true, predictions_test)

# 输出评估指标
print(f"RMSE: {rmse:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R2: {R2:.4f}")
print(f"SMAPE: {smape:.2f}%")

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(predictions_test[:, 0], label='预测值 (输出1)')
plt.plot(Y_test_true[:, 0], label='真实值 (输出1)', linestyle='dashed')
plt.xlabel('样本索引')
plt.ylabel('预测值')
plt.title('LSTM 回归模型预测结果')
plt.legend()
plt.show()

# 定义保存路径
save_path = "E:\\python3.8-study\\figure\\"

# 分别保存为 Excel 文件
pred_file_path = save_path + "LSTM_test_pre.xlsx"
true_file_path = save_path + "LSTM_test_true.xlsx"

pd.DataFrame(predictions_test, columns=['杨氏模量_预测', '粘聚力_预测', '摩擦角_预测', '抗拉强度_预测']).to_excel(pred_file_path, index=False)
pd.DataFrame(Y_test_true, columns=['杨氏模量_真实', '粘聚力_真实', '摩擦角_真实', '抗拉强度_真实']).to_excel(true_file_path, index=False)

# 处理新的数据集
data_bd = pd.read_excel('E:\\python3.8-study\\biaodinglunwen.xlsx')
bd = data_bd.iloc[:, :X.shape[1]].values  # 确保输入特征数匹配
bd_normalized = scaler_X.transform(bd)
bd_normalized = reshape_for_lstm(bd_normalized)  # 确保形状正确

# 进行预测
predictions_bd_normalized = grid_result.best_estimator_.predict(bd_normalized)
predictions_bd = scaler_Y.inverse_transform(predictions_bd_normalized.reshape(-1, Y.shape[1]))

# 输出预测结果
print(predictions_bd)