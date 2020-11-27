import numpy as np
from dqn_agent import DQNAgent
from utils import make_env, plot_learning_curve
import time



ENV = 'PongNoFrameskip-v4'
NUM_GAMES = 1000
MEM_SIZE = 96000
BATCH_SIZE = 32
LR = .0001
GAMMA = .99
EPSILON = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 5e-6
ALGORITHM = 'DQNAgent'
POLICY_UPDATE = 1000
CHECKPOINT_DIR = 'models/'
FIGURES_DIR = 'figures/'
MOVMEAN = 100


if __name__ == "__main__":
    env = make_env(ENV)
    best_score = -np.inf
    load_checkpoint = False
    agent = DQNAgent(gamma=GAMMA, epsilon=EPSILON, lr=LR, input_dims=env.observation_space.shape,
                     n_actions=env.action_space.n, mem_size=MEM_SIZE, eps_min=EPSILON_MIN, batch_size=BATCH_SIZE,
                     replace_count=POLICY_UPDATE, eps_dec=EPSILON_DECAY, checkpoint_dir=CHECKPOINT_DIR, algorithm=ALGORITHM,
                     env_name=ENV)

    if load_checkpoint:
        agent.load_models()

    file_name = agent.algorithm + "_" + agent.env_name + "_lr-" + str(agent.lr) + "_" + str(NUM_GAMES) + "-games"
    figure_file = FIGURES_DIR + file_name + ".png"

    n_steps = 0
    scores = []
    eps_hist = []
    steps_arr = []

    tic = time.perf_counter()
    tic1 = time.perf_counter()
    for i in range(NUM_GAMES):
        done = False
        score = 0
        obs = env.reset()
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            score += reward

            # if training store the transition
            if not load_checkpoint:
                agent.store_transition(obs, action, reward, next_obs, int(done))
                agent.learn()
            obs = next_obs
            n_steps += 1
        scores.append(score)
        steps_arr.append(n_steps)
        eps_hist.append(agent.epsilon)

        avg = np.mean(scores[-MOVMEAN:])
        if avg > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg

        plot_learning_curve(steps_arr, scores, eps_hist, figure_file)

        if i % 10 == 0:
            toc1 = time.perf_counter()
            print("episode: {}\tscore: {:.1f}\tavg score: {:.1f}\tbest score: {:.1f}\teps: {:.2f}\tsteps: {}\ttime: {:.3f}".format(i, score, avg, best_score, agent.epsilon, n_steps, (toc1-tic1)))
            tic1 = time.perf_counter()
    toc
    print("elapsed training time: {:.3f}".format((toc-tic)))

