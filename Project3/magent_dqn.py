#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
from collections import deque, namedtuple
from itertools import count
from typing import List

import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from agent import Agent
from dqn_model import DQN
from environment import Environment

writer = SummaryWriter()

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

# Checkout recommended values at the end of the original paper
EPISODES = 1000
LEARNING_RATE = 1.5e-4  # alpha
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 10000
EPSILON = 1.0
EPSILON_END = 0.025
FINAL_EXPL_FRAME = 1000000
TARGET_UPDATE_FREQUENCY = 1000
SAVE_MODEL_AFTER = 5000
DECAY_EPSILON_AFTER = 10000
# decay_per_step = (self.epsilon - epsilon_min) / no_of_steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device is:', device)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
rew_buffer = deque([0.0], maxlen=100)


class ReplayBuffer(object):
    def __init__(self, capacity: int):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        """Save a transition"""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        """Randomly sample 'batch_size' number of transitions from buffer"""
        # samples = random.sample(self.buffer, batch_size)
        # for elem in samples:
        #     self.buffer.remove(elem)
        # return samples

        samples = []
        for _ in range(batch_size):
            index = random.randrange(len(self.buffer))
            samples.append(self.buffer[index])
            del self.buffer[index]
        return samples

    def __len__(self):
        return len(self.buffer)


class Agent_DQN(Agent):
    """
    Initialize everything you need here.
    For example:
        parameters for neural network
        initialize Q net and target Q net
        parameters for replay buffer
        parameters for q-learning; decaying epsilon-greedy
        ...
    """
    def __init__(self, env: Environment, args):
        super(Agent_DQN, self).__init__(env)
        self.env = env
        self.action_count = self.env.action_space.n
        in_channels = 4  # (R, G, B, Alpha)

        self.Q_net = DQN(in_channels, self.action_count).to(device)
        self.target_Q_net = DQN(in_channels, self.action_count).to(device)
        self.target_Q_net.load_state_dict(self.Q_net.state_dict())
        self.optimizer = optim.Adam(self.Q_net.parameters(), lr=LEARNING_RATE)

        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.training_steps = 0

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
        pass

    def init_game_setting(self):
        """
        Testing function will call this function at the beginning of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        pass

    def make_action(self, observation: np.ndarray, test: bool =True) -> int:
        """
        Returns predicted action of your agent from trained model
        """
        # Get observation in correct format for network
        state = self.format_state(observation)

        # Get Q from network/model
        Q = self.Q_net.forward(state)

        # Greedy/deterministic action
        max_q_index = torch.argmax(Q, dim=1)[0]

        return max_q_index.detach().item()

    def get_eps_greedy_action(self, greedy_action: int, epsilon: float):
        """
        Take the deterministic action given by the network and return an epsilon
        greedy action.
        """
        probability = np.ones(self.action_count) * epsilon / self.action_count  # exploration
        probability[greedy_action] += 1 - epsilon  # exploitation
        return np.random.choice(np.arange(self.action_count), p=probability)

    def train(self, no_of_episodes: int = EPISODES):
        """
        """
        for epi_num in range(no_of_episodes):  # One episode is one complete game
            print("Episode", epi_num)
            episode_reward = 0
            curr_state = self.env.reset()

            for step in count():
                if epi_num > DECAY_EPSILON_AFTER:
                    epsilon = np.interp(step, [0, FINAL_EXPL_FRAME], [EPSILON, EPSILON_END])
                else:
                    epsilon = EPSILON
                action = self.get_eps_greedy_action(self.make_action(curr_state), epsilon)
                next_state, reward, done, info, _ = self.env.step(action)

                # Convert numpy arrays/int to tensors
                curr_state_t = self.format_state(curr_state)
                next_state_t = self.format_state(next_state)
                action_t = torch.tensor([action], device=device)
                reward_t = torch.tensor([reward], device=device)

                self.buffer.push(curr_state_t, action_t, reward_t, next_state_t)

                curr_state = next_state
                episode_reward += reward

                # Optimize
                self.optimize_model()

                if done:
                    rew_buffer.append(episode_reward)
                    break

            # Logging
            writer.add_scalar("Epsilon vs Step", epsilon, step)
            if epi_num % 100 == 0:
                writer.add_scalar("Mean reward(100) vs Episode", np.mean(rew_buffer), epi_num)
                writer.add_scalar(
                    "Mean reward(100) vs Training steps",
                    np.mean(rew_buffer),
                    self.training_steps
                )

            if epi_num % TARGET_UPDATE_FREQUENCY == 0:
                self.target_Q_net.load_state_dict(self.Q_net.state_dict())

            if epi_num % SAVE_MODEL_AFTER == 0:
                torch.save(self.Q_net.state_dict(), "vanilla_dqn_model.pth")

        torch.save(self.Q_net.state_dict(), "vanilla_dqn_model.pth")
        print("Complete")
        writer.flush()
        writer.close()

    def optimize_model(self) -> None:
        """
        """
        if len(self.buffer) < BUFFER_SIZE:
            return

        self.training_steps += 1

        transitions = self.buffer.sample(BATCH_SIZE)

        # Convert batch array of transitions to Transition of batch arrays
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_terminal_next_state_batch = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        sav_t = self.Q_net(state_batch)
        state_action_values = sav_t[torch.arange(sav_t.size(0)), action_batch]

        # Get state-action values
        non_terminal_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool
        )
        next_state_Q_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_Q_values[non_terminal_mask] = self.target_Q_net(
            non_terminal_next_state_batch
        ).max(1)[0].detach()

        # Compute the ground truth
        ground_truth_q_values = reward_batch + GAMMA*next_state_Q_values
        ground_truth_q_values = torch.reshape(
            ground_truth_q_values.unsqueeze(1),
            (1, BATCH_SIZE)
        )[0]

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, ground_truth_q_values)

        # Optimize the model
        self.optimizer.zero_grad(set_to_none=True)  # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
        loss.backward()
        for param in self.Q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    @staticmethod
    def format_state(state: np.ndarray) -> torch.Tensor:
        """
        """
        state = np.asarray(state, dtype=np.float32) / 255

        # Transpose into torch order (CHW)
        state = state.transpose(2, 0, 1)

        # Add a batch dimension (BCHW)
        return torch.from_numpy(state).unsqueeze(0)