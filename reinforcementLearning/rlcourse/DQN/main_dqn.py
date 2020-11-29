import numpy as np
import dqn_agents as Agents
from utils import make_env, plot_learning_curve
import warnings
import time
import argparse
import os
warnings.filterwarnings(action='once')



# ENV = 'PongNoFrameskip-v4'
# #ENV = 'CartPole-v1'
# NUM_GAMES = 1000
# MEM_SIZE = 96000
# BATCH_SIZE = 32
# LR = .0001
# GAMMA = .99
# EPSILON = 1.0
# EPSILON_MIN = 0.1
# EPSILON_DECAY = 1e-5
# ALGORITHM = 'DQNAgent'
# POLICY_UPDATE = 1000
# CHECKPOINT_DIR = 'models/'
FIGURES_DIR = 'figures/'
MOVMEAN = 100


def train(env, agent, n_games, load_checkpoint=False, early_stopping=False):
    best_score = -np.inf
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    file_name = agent.algorithm + "_" + agent.env_name + "_lr-" + str(agent.lr) + "_" + str(n_games) + "-games_batch-size-" + str(agent.batch_size)
    figure_file = FIGURES_DIR + file_name + ".png"

    n_steps = 0
    scores = []
    eps_hist = []
    steps_arr = []

    tic = time.perf_counter()
    tic1 = time.perf_counter()
    for i in range(n_games):
        done = False
        score = 0
        obs = env.reset()
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            #env.render()
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
    toc = time.perf_counter()
    print("elapsed training time: {:.3f}".format((toc-tic)))


def play(env, agent, n_games, frame_delay):
    agent.load_models()
    for i in range(n_games):
        done = False
        score = 0
        obs = env.reset()
        while not done:
            action = agent.choose_action(obs, network="policy")
            next_obs, reward, done, info = env.step(action)
            env.render()
            score += reward
            obs = next_obs
            time.sleep(frame_delay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Learning Human Level Control Paper")
    parser.add_argument('-n_games', type=int, default=101, help="number of games to play")
    parser.add_argument('-lr', type=float, default=1e-4, help="learning rate for optimizer")
    parser.add_argument('-epsilon', type=float, default=1.0, help="initial epsilon-greedy value")
    parser.add_argument('-eps_dec', type=float, default=1e-5, help="linear decay factor")
    parser.add_argument('-eps_min', type=float, default=.1, help="minimum epsilon-greedy value")
    parser.add_argument('-gamma', type=float, default=.99, help="discount factor Q value update")
    parser.add_argument('-max_mem', type=int, default=64000, help="replay buffer size")
    parser.add_argument('-batch_size', type=int, default=32, help="batch size for replay sampling")
    parser.add_argument('-frame_skip', type=int, default=4, help="number of frames to stack")
    parser.add_argument('-update', type=int, default=500, help="interval for updating policy network")
    parser.add_argument('-env', type=str, help="Atari Environment.\nPongNoFrameskip-v4\n"
                                                             "BreakoutNoFrameskip-v4\n"
                                                             "SpaceInvadersNoFrameskip-v4\n"
                                                             "EnduroNoFrameskip-v4\n"
                                                             "AtlantisNoFrameskip-v4")
    parser.add_argument('-gpu', type=str, default="0", help='GPU: 0 or 1')
    parser.add_argument('-mode', type=str, default="train", help="train or play")
    parser.add_argument('-load_checkpoint', type=bool, default=False, help="load model checkpoint (path argument required if True)")
    parser.add_argument('-path', type=str, default='models/', help="path for model saving/loading")
    parser.add_argument('-agent', type=str, default="DQNAgent", help="DQNAgent/DDQNAgent/DuelingDDQNAgent")
    parser.add_argument('-fire_first', type=bool, default=False, help="if playing SpaceInvadersNoFrameskip-v4, fire first?")
    parser.add_argument('-frame_delay', type=float, default=.05, help="frame delay in seconds, default=.05")
    parser.add_argument('-early_stopping', type=bool, default=False, help="stop training early when performance saturates")
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    env = make_env(args.env, repeat=args.frame_skip, fire_first=args.fire_first)
    user_agent = getattr(Agents, args.agent)
    agent = user_agent(gamma=args.gamma, epsilon=args.epsilon, lr=args.lr,
                     input_dims=env.observation_space.shape, n_actions=env.action_space.n,
                     mem_size=args.max_mem, eps_min=args.eps_min, batch_size=args.batch_size,
                    replace_count=args.update, eps_dec=args.eps_dec,
                     checkpoint_dir=args.path, algorithm=args.agent,
                    env_name=args.env)

    if(args.mode == "train"):
        print("training...")
        train(env, agent, args.n_games, args.early_stopping)
    elif(args.mode == "play"):
        print("playing...")
        play(env, agent, args.n_games, args.frame_delay)

