import gymnasium as gym
import highway_env
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder


# 定义虚拟环境
class VirtualHighwayEnv(gym.Env):
    def __init__(self, env, model, scaler_X, scaler_Y):
        super(VirtualHighwayEnv, self).__init__()
        self.env = env
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.current_state = None

    def reset(self):
        self.current_state, _ = self.env.reset()
        return self.current_state

    def step(self, action):
        X = np.hstack((self.current_state.reshape(1, -1), action.reshape(1, -1)))
        X_scaled = self.scaler_X.transform(X)
        next_state_scaled = self.model.predict(X_scaled)
        next_state = self.scaler_Y.inverse_transform(next_state_scaled).reshape(self.current_state.shape)

        # 在虚拟环境中无法计算真实的奖励和终止条件，所以需要模拟
        reward = -np.sum((next_state - self.current_state) ** 2)  # 示例奖励函数
        self.current_state = next_state
        done = False  # 示例终止条件
        return next_state, reward, done, {}

    def render(self, mode='human'):
        self.env.render(mode)

    def close(self):
        self.env.close()


# 创建和配置环境
#env = DummyVecEnv([lambda: gym.make("highway-v0",render_mode = 'rgb_array')])
env = gym.make("highway-v0",render_mode = 'rgb_array')
env.unwrapped.configure({
    "action": {
        "type": "ContinuousAction"  # 使用连续动作空间
    },
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 100,
        "features": ["x", "y", "vx", "vy"],
        "normalize": False
    },
    "policy_frequency": 15,
    "duration": 20
})
#video_length = 2 * env.envs[0].config["duration"]
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
data = collect_data(env, 10000)
observations = np.array([d[0] for d in data])
actions = np.array([d[1] for d in data]).reshape(-1, env.action_space.shape[0])
next_observations = np.array([d[2] for d in data])

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
model = MLPRegressor(hidden_layer_sizes=(1000, 1000), max_iter=10000)
model.fit(X_train, Y_train)

# 预测并计算均方误差
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
print(f"MLP Regressor Mean Squared Error: {mse}")


# 定义遗传算法的评估函数
def get_evaluate_function(env, model, scaler_X, scaler_Y):
    def evaluate(individual):
        virtual_env = VirtualHighwayEnv(env, model, scaler_X, scaler_Y)
        obs = virtual_env.reset()
        total_reward = 0
        for _ in range(1000):
            action = np.array(individual)  # 将个体作为动作
            obs, reward, done, _ = virtual_env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward,

    return evaluate


# 设置遗传算法
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, low=-1.0, high=1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=env.action_space.shape[0])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", get_evaluate_function(env, model, scaler_X, scaler_Y))

# 生成初始种群
population = toolbox.population(n=500)

# 定义遗传算法的参数
NGEN = 1000  # 迭代次数
CXPB = 0.5  # 交叉概率
MUTPB = 0.2  # 变异概率

# 运行遗传算法
for gen in range(NGEN):
    # 评估所有个体的适应度
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # 选择下一代的个体
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # 应用交叉和变异
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if np.random.rand() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if np.random.rand() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # 评估新个体的适应度
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # 生成新的种群
    population[:] = offspring

    # 记录并打印最优个体
    fits = [ind.fitness.values[0] for ind in population]
    length = len(population)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print(f"Generation {gen}: Max {max(fits)}, Avg {mean}, Std {std}")

best_ind = tools.selBest(population, 1)[0]
print("Best individual is:", best_ind)
print("With fitness:", best_ind.fitness.values)

#video_length = 2 * env.envs[0].config["duration"]

# 在真实环境中评估训练好的策略
def evaluate_real_env(env, individual):
    obs, _ = env.reset()
    total_reward = 0


    #env = DummyVecEnv(model)
    #video_length = 2 * env.envs[0].config["duration"]
    #obs, info = env.reset()



    for _ in range(1000):
        action = np.array(individual)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()
        if done:
            break
    return total_reward


real_env_reward = evaluate_real_env(env, best_ind)
print("Total reward in real environment:", real_env_reward)

env.close()
