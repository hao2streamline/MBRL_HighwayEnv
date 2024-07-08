import numpy as np
import policy
from sklearn.ensemble import RandomForestRegressor

class ModelBasedRL:
    def __init__(self, env):
        self.env = env
        self.model = None
        self.experience = []

    def collect_data(self, policy, episodes=10):
        for _ in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = policy(state)
                next_state, reward, done, _ = self.env.step(action)
                self.experience.append((state, action, reward, next_state))
                state = next_state


