import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_idx = 0
        self.state_mem = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.next_state_mem = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_mem = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_mem = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, current_state, action, reward, next_state, done):
        idx = self.mem_idx % self.mem_size
        self.state_mem[idx] = current_state
        self.action_mem[idx] = action
        self.next_state_mem[idx] = next_state
        self.reward_mem[idx] = reward
        self.terminal_mem[idx] = done
        self.mem_idx += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_idx, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        next_states = self.next_state_mem[batch]
        dones = self.terminal_mem[batch]

        return states, actions, rewards, next_states, dones