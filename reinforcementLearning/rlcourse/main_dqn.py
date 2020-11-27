import numpy as np
from dqn_agent import DQNAgent
from utils import make_env, plot_learning_curve

if __name__ == "__main__":
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
