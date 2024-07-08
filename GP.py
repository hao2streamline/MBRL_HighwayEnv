import gymnasium as gym
import highway_env
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
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
        "normalize": True
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
        #print(f'obs: {obs} \naction: {action} \nnext_obs: {next_obs}')
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

# 创建并训练高斯过程模型
kernel = C(1.0, (1e-3, 1e5)) * RBF(1.0, (1e-2, 1e2))
gp_models = []

for i in range(Y_train.shape[1]):
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, optimizer='fmin_l_bfgs_b', alpha=1e-10)
    gp.fit(X_train, Y_train[:, i])
    gp_models.append(gp)

# 使用测试数据进行预测
Y_pred_scaled = np.hstack([model.predict(X_test).reshape(-1, 1) for model in gp_models])
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

# 计算均方误差
mse = mean_squared_error(scaler_Y.inverse_transform(Y_test), Y_pred)
print(f"Mean Squared Error: {mse}")

# 关闭环境
env.close()
