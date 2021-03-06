import numpy as np
from dqn import DQN, DuelingDDQN
from replay_memory import ReplayBuffer
import torch as T

class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, eps_min=.01, eps_dec=5e-7,
                 replace_count=1000, algorithm=None, env_name=None, checkpoint_dir='/checkpoints'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_count = replace_count
        self.algorithm = algorithm
        self.env_name = env_name
        self.checkpoint_dir = checkpoint_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        print(type(self).__name__)
        self.q_eval = object
        self.q_policy = object

    def store_transition(self, current_state, action, reward, next_state, done):
        self.memory.store_transition(current_state, action, reward, next_state, done)

    def sample_memory(self):
        current_state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        current_states = T.tensor(current_state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        next_states = T.tensor(next_state).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        return current_states, actions, rewards, next_states, dones

    def update_policy_network(self):
        if(self.learn_step_counter % self.replace_count == 0):
            self.q_policy.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_policy.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_policy.load_checkpoint()

    def pre_learn(self):
        if self.memory.mem_idx < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()
        self.update_policy_network()
        current_states, actions, rewards, next_states, dones = self.sample_memory()
        indices = np.arange(self.batch_size)
        return current_states, actions, rewards, next_states, dones, indices

    def post_learn(self, q_target, q_pred):
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def choose_action(self, observation, network):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError


class DQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = DQN(self.lr, self.n_actions, input_dims=self.input_dims,
                          name=self.env_name+"_"+self.algorithm+"_q_eval", checkpoint_dir=self.checkpoint_dir)

        self.q_policy = DQN(self.lr, self.n_actions, input_dims=self.input_dims,
                          name=self.env_name+"_"+self.algorithm+"_q_policy", checkpoint_dir=self.checkpoint_dir)

    def choose_action(self, observation, network="eval"):
        if(network == "eval"):
            r = np.random.random()
            if r > self.epsilon:
                state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
                actions = self.q_eval.forward(state)
            else:
                action = np.random.choice(self.action_space)
                return action
        elif(network == "policy"):
            state = T.tensor([observation], dtype=T.float).to(self.q_policy.device)
            actions = self.q_policy.forward(state)
        action = T.argmax(actions).item()

        return action

    def learn(self):
        if self.memory.mem_idx < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()
        self.update_policy_network()
        current_states, actions, rewards, next_states, dones = self.sample_memory()
        indices = np.arange(self.batch_size)
        #current_states, actions, rewards, next_states, dones, indices = self.pre_learn()
        q_pred = self.q_eval.forward(current_states)[indices, actions]
        q_policy = self.q_policy.forward(next_states).max(dim=1)[0] # get just values
        q_policy[dones.bool()] = 0.0

        q_target = rewards + self.gamma*q_policy
        #self.post_learn(q_target, q_pred)
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()


class DDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DDQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = DQN(self.lr, self.n_actions, input_dims=self.input_dims,
                          name=self.env_name+"_"+self.algorithm+"_q_eval", checkpoint_dir=self.checkpoint_dir)

        self.q_policy = DQN(self.lr, self.n_actions, input_dims=self.input_dims,
                          name=self.env_name+"_"+self.algorithm+"_q_policy", checkpoint_dir=self.checkpoint_dir)

    def choose_action(self, observation, network="eval"):
        if(network == "eval"):
            r = np.random.random()
            if r > self.epsilon:
                state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
                actions = self.q_eval.forward(state)
            else:
                action = np.random.choice(self.action_space)
                return action
        elif(network == "policy"):
            state = T.tensor([observation], dtype=T.float).to(self.q_policy.device)
            actions = self.q_policy.forward(state)
        action = T.argmax(actions).item()

        return action

    def learn(self):
        if self.memory.mem_idx < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()
        self.update_policy_network()
        current_states, actions, rewards, next_states, dones = self.sample_memory()
        indices = np.arange(self.batch_size)
        #current_states, actions, rewards, next_states, dones, indices = self.pre_learn()
        q_pred = self.q_eval.forward(current_states)[indices, actions]
        q_policy = self.q_policy.forward(next_states)
        q_eval = self.q_eval.forward(next_states)

        max_actions = T.argmax(q_eval, dim=1)
        q_policy[dones.bool()] = 0.0
        q_target = rewards + self.gamma*q_policy[indices, max_actions]
        #self.post_learn(q_target, q_pred)
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()


class DuelingDDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DuelingDDQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = DuelingDDQN(self.lr, self.n_actions, input_dims=self.input_dims,
                          name=self.env_name+"_"+self.algorithm+"_q_eval", checkpoint_dir=self.checkpoint_dir)

        self.q_policy = DuelingDDQN(self.lr, self.n_actions, input_dims=self.input_dims,
                          name=self.env_name+"_"+self.algorithm+"_q_policy", checkpoint_dir=self.checkpoint_dir)

    def choose_action(self, observation, network="eval"):
        if(network == "eval"):
            r = np.random.random()
            if r > self.epsilon:
                state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
                _, advantage = self.q_eval.forward(state)
            else:
                action = np.random.choice(self.action_space)
                return action
        elif(network == "policy"):
            state = T.tensor([observation], dtype=T.float).to(self.q_policy.device)
            _, advantage = self.q_policy.forward(state)
        action = T.argmax(advantage).item()
        return action

    def learn(self):
        if self.memory.mem_idx < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()
        self.update_policy_network()
        current_states, actions, rewards, next_states, dones = self.sample_memory()
        indices = np.arange(self.batch_size)
        # current_states, actions, rewards, next_states, dones, indices = self.pre_learn()
        Vs, As, = self.q_eval.forward(current_states)
        Vs_policy, As_policy = self.q_policy.forward(next_states)

        Vs_eval, As_eval = self.q_eval.forward(next_states)

        q_pred = T.add(Vs, (As - As.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(Vs_policy, (As_policy - As_policy.mean(dim=1, keepdim=True)))

        q_eval = T.add(Vs_eval, (As_eval - As_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones.bool()] = 0.0
        q_target = rewards + self.gamma*q_next[indices, max_actions]
        #self.post_learn(q_target, q_pred)
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
