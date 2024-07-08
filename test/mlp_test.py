import gymnasium as gym
import MLP
import evolution
import highway_env
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

def train_env():
    env = gym.make("highway-fast-v0",render_mode = 'rgb_array')
    env.configure(
        {
            "action": {
                "type": "ContinuousAction"  # 使用连续动作空间
            },
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 10,
                "features": ["x", "y", "vx", "vy"],
                "normalize": False
            },
        "policy_frequency": 15,
        "duration": 20
        }
    )
    env.reset()
    return env


def test_env():
    env = train_env()
    env.configure({"policy_frequency": 15, "duration": 20})
    env.reset()
    return env

if __name__ == "__main__":
    # Train
    model = MLP.train(env=train_env())
    #model.learn(total_timesteps=int(1e5))
    MLP.save_model(model, filepath="highway_mlp/model.pkl")

    # Record video
    model = MLP.load_model("highway_mlp/model.pkl")

    env = DummyVecEnv([test_env])
    video_length = 2 * env.envs[0].config["duration"]
    env = VecVideoRecorder(
        env,
        "videos/",
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="mlp-agent",
    )
    obs, info = env.reset()

    individual = evolution.evo()
    def evaluate_real_env(env, individual):
        obs, _ = env.reset()
        total_reward = 0

       #individual = evolution.evo()

        for _ in range(video_length + 1):
            action = np.array(individual)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            env.render()
            if done:
                break
        return total_reward
        #env.close()


    real_env_reward = evaluate_real_env(env, individual)
    print("Total reward in real environment:", real_env_reward)

    env.close()