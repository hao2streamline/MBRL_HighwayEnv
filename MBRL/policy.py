import gymnasium as gym
import highway_env
import numpy as np
from deap import base, creator, tools, algorithms
import Env


def get_evaluate_function(env, model, scaler_X, scaler_Y):
    def evaluate(individual):
        virtual_env = Env.VirtualHighwayEnv(env, model, scaler_X, scaler_Y)
        obs = virtual_env.reset()
        total_reward = 0
        for _ in range(100):
            action = np.array(individual)  # 将个体作为动作
            obs, reward, done, info = virtual_env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward,

    return evaluate

def evo(env, model, scaler_X, scaler_Y):
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
    population = toolbox.population(n=50)

    # 定义遗传算法的参数
    NGEN = 40  # 迭代次数
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

    return best_ind
