import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from geneticalgorithm import geneticalgorithm as ga

# 加载数据
data = pd.read_excel('E:\\python3.8-study\\shuju3.0.xlsx')

# 提取输入（特征）和输出（目标）
X = data.iloc[:, 4:9].values  # 5个输入特征
Y = data.iloc[:, 3].values.reshape(-1, 1)  # 只选择第一列作为输出
print(Y)
# 数据归一化
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)
Y_normalized = scaler_Y.fit_transform(Y)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y_normalized, test_size=0.3, random_state=42)

# 调整 LSTM 输入形状
X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# 定义 LSTM 模型
def create_lstm_model(learning_rate=0.001, lstm_units=16, dense_units=64):
    model = Sequential([
        LSTM(lstm_units, activation='tanh', input_shape=(1, 5)),  # 5 维输入
        Dense(dense_units, activation='relu'),
        Dense(1, activation='linear')  # 修改为单输出
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# 训练 LSTM
lstm_model = KerasRegressor(model=create_lstm_model, epochs=300, batch_size=64, verbose=0)
lstm_model.fit(X_train_lstm, Y_train)

# 训练基模型
rf_model = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=42)
rf_model.fit(X_train, Y_train.ravel())  # RandomForest 需要 1D 目标

xgb_model = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
xgb_model.fit(X_train, Y_train.ravel())  # XGBRegressor 也需要 1D 目标

# 获取基模型预测结果
lstm_pred_train = lstm_model.predict(X_train_lstm).reshape(-1, 1)
rf_pred_train = rf_model.predict(X_train).reshape(-1, 1)
xgb_pred_train = xgb_model.predict(X_train).reshape(-1, 1)

# 组合基模型预测结果
ensemble_input_train = np.hstack((lstm_pred_train, rf_pred_train, xgb_pred_train))

# 遗传算法优化 BP 神经网络
def fitness_function(params):
    neurons_per_layer = int(params[0])
    learning_rate = params[1]
    batch_size = int(params[2])
    hidden_layers = int(params[3])
    activation_idx = int(params[4])
    optimizer_idx = int(params[5])

    activation_options = ['tanh', 'relu', 'sigmoid']
    optimizer_options = ['adam', 'sgd', 'rmsprop']

    activation = activation_options[activation_idx]
    optimizer = optimizer_options[optimizer_idx]

    model = create_bp_model(neurons_per_layer=neurons_per_layer,
                            learning_rate=learning_rate,
                            activation=activation,
                            hidden_layers=hidden_layers,
                            optimizer=optimizer)

    model.fit(ensemble_input_train, Y_train, epochs=100, batch_size=batch_size, verbose=0)
    pred = model.predict(ensemble_input_train)
    mse = mean_squared_error(Y_train, pred)
    return mse

# 定义 BP 神经网络
def create_bp_model(neurons_per_layer=30, learning_rate=0.0003, activation='tanh', hidden_layers=3, optimizer='adam'):
    model = Sequential()
    model.add(Dense(neurons_per_layer, input_dim=3, activation=activation))  # 输入维度变为3
    for _ in range(hidden_layers):
        model.add(Dense(neurons_per_layer, activation=activation))
    model.add(Dense(1, activation='linear'))  # 只输出1个值

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

# 定义超参数搜索范围
varbound = np.array([
    [16, 128],  # neurons_per_layer
    [0.00001, 0.01],  # learning_rate
    [16, 64],  # batch_size
    [1, 5],  # hidden_layers
    [0, 2],  # activation
    [0, 2]  # optimizer
])

# 运行遗传算法
algorithm_param = {
    'max_num_iteration': 200,
    'population_size': 20,
    'mutation_probability': 0.01,
    'parents_portion': 0.3,
    'crossover_probability': 0.8,
    'elit_ratio': 0.1,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': 25
}

model_ga = ga(
    function=fitness_function,
    dimension=6,
    variable_type='real',
    variable_boundaries=varbound,
    algorithm_parameters=algorithm_param
)

model_ga.run()
best_params = model_ga.output_dict['variable']

# 解析最佳参数
best_neurons = int(best_params[0])
best_learning_rate = best_params[1]
best_batch_size = int(best_params[2])
best_hidden_layers = int(best_params[3])
best_activation = ['tanh', 'relu', 'sigmoid'][int(best_params[4])]
best_optimizer = ['adam', 'sgd', 'rmsprop'][int(best_params[5])]

# 训练最终 BP 神经网络
bp_model = create_bp_model(neurons_per_layer=best_neurons,
                           learning_rate=best_learning_rate,
                           activation=best_activation,
                           hidden_layers=best_hidden_layers,
                           optimizer=best_optimizer)
bp_model.fit(ensemble_input_train, Y_train, epochs=270, batch_size=best_batch_size, verbose=0)

# 测试集预测
ensemble_test_input = np.hstack((
    lstm_model.predict(X_test_lstm).reshape(-1, 1),
    rf_model.predict(X_test).reshape(-1, 1),
    xgb_model.predict(X_test).reshape(-1, 1)
))

predictions_normalized = bp_model.predict(ensemble_test_input)
predictions = scaler_Y.inverse_transform(predictions_normalized)
Y_test_re = scaler_Y.inverse_transform(Y_test)

# 计算误差指标
rmse = np.sqrt(mean_squared_error(Y_test_re, predictions))
mae = mean_absolute_error(Y_test_re, predictions)
r2 = r2_score(Y_test_re, predictions)

print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
