import gymnasium as gym
import highway_env
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 创建和配置环境
env = gym.make("highway-v0")
env.configure({
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["x", "y", "vx", "vy"],
        "normalize": False
    },
    "policy_frequency": 15
})
env.reset()

# 使用PPO算法训练策略
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 收集数据以训练环境模型
def collect_data(env, model, steps):
    data = []
    obs, _ = env.reset()
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        data.append((obs, action, next_obs))
        obs = next_obs
        if done:
            obs, _ = env.reset()
    return data

data = collect_data(env, model, 1000)
observations = np.array([d[0] for d in data])
actions = np.array([d[1] for d in data])
next_observations = np.array([d[2] for d in data])

# 确保所有观测值形状一致
obs_shape = observations[0].shape
observations = np.array([np.reshape(d[0], obs_shape) for d in data])
next_observations = np.array([np.reshape(d[2], obs_shape) for d in data])



# 准备训练数据
X = np.hstack((observations, actions))

print('x:',X)
Y = next_observations

# 分割训练和测试数据
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 创建并训练高斯过程模型
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# 对于每个输出维度训练一个GP模型
models = []
for i in range(Y_train.shape[1]):
    gp.fit(X_train, Y_train[:, i])
    models.append(gp)

# 使用测试数据进行预测
Y_pred = np.hstack([model.predict(X_test).reshape(-1, 1) for model in models])

# 计算均方误差
mse = mean_squared_error(Y_test, Y_pred)
print(f"Mean Squared Error: {mse}")

# 关闭环境
env.close()
