import copy
import random
import torch
import numpy as np
from gym import make
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque

from buffer import Buffer


LEARNING_RATE = 5e-4
CONST_VAL = 0
GAMMA = 0.99
STEPS_PER_UPDATE = 4
BATCH_SIZE = 128
STATE_UNITS = 256
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000

class DQN:
    def __init__(self, state_dim, action_dim):
        self.steps = CONST_VAL # Do not change
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        # defining the buffer
        self.buffer = Buffer()
        self.model = nn.Sequential(
            nn.Linear(state_dim, STATE_UNITS),
            nn.ReLU(),
            nn.Linear(STATE_UNITS, STATE_UNITS),
            nn.ReLU(),
            nn.Linear(STATE_UNITS, STATE_UNITS),
            nn.ReLU(),
            nn.Linear(STATE_UNITS, action_dim),
        ) # Torch model
        # device run
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # detected target
        self.target = copy.deepcopy(self.model).to(self.device)
        # optimizer
        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
        # loss mse
        self.loss = nn.MSELoss()

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.buffer.add(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        batch = self.buffer.sample(self.batch_size)
        return list(zip(*batch))
        
    def train_step(self, batch):
        # Use batch to update DQN's network.
        state, action, next_state, reward, done = batch
        state = torch.tensor(np.array(state), dtype=torch.float32)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32)
        reward = torch.tensor(np.array(reward), dtype=torch.float32).view(-1)
        done = torch.tensor(np.array(done), dtype=torch.bool)
        action = torch.tensor(np.array(action), dtype=torch.int64).view(-1, 1)
        # target network            
        with torch.no_grad():
            q_target = self.target(next_state).max(dim=-1)[0]
            q_target[done] = 0
            q_target = reward + self.gamma * q_target
        q_func = self.model(state).gather(1, action.reshape(-1, 1))
        # calculate loss
        loss = self.loss(q_func, q_target.unsqueeze(1))
        # step
        self.optimizer.zero_grad()
        # calculate loss
        loss.backward()
        # step optimizer
        self.optimizer.step()
        
    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or 
        # assign a values of network parameters via PyTorch methods.
        self.target = copy.deepcopy(self.model)

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = np.array(state)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            action = self.model(state).numpy()
        return np.argmax(action)

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model, "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns

if __name__ == "__main__":
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    state = env.reset()
    
    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
    
    for i in range(TRANSITIONS):
        #Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
        if (i + 1) % (TRANSITIONS//100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            dqn.save()
