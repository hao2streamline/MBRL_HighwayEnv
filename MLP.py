import gymnasium as gym
import highway_env
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# 创建和配置环境
env = gym.make("highway-v0")
env.unwrapped.configure({
    "action": {
        "type": "ContinuousAction"  # 使用连续动作空间
    },
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["x", "y", "vx", "vy"],
        "normalize": False
    },
    "policy_frequency": 15
})
obs, _ = env.reset()

# 数据收集函数
def collect_data(env, steps):
    data = []
    obs, _ = env.reset()
    for _ in range(steps):
        action = env.action_space.sample()  # 随机动作
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        data.append((obs, action, next_obs))
        obs = next_obs
        if done:
            obs, _ = env.reset()
    return data

# 收集数据
data = collect_data(env, 1000)
observations = np.array([d[0] for d in data])
actions = np.array([d[1] for d in data]).reshape(-1, env.action_space.shape[0])  # 确保actions是二维数组
next_observations = np.array([d[2] for d in data])

# 确保所有观测值形状一致
obs_shape = observations[0].shape
observations = np.array([np.reshape(d[0], obs_shape) for d in data])
next_observations = np.array([np.reshape(d[2], obs_shape) for d in data])

# 准备训练数据
X = np.hstack((observations.reshape(observations.shape[0], -1), actions))
Y = next_observations.reshape(next_observations.shape[0], -1)

# 数据标准化
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

# 分割训练和测试数据
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# 训练多层感知机模型
model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=10000)
model.fit(X_train, Y_train)

# 预测并计算均方误差
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
print(f"MLP Regressor Mean Squared Error: {mse}")

# 关闭环境
env.close()
