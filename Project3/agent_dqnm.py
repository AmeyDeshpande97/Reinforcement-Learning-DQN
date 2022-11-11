#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque, namedtuple
import os
import sys
from itertools import count
import torch
import torch.nn.functional as F
import torch.optim as optim
from environment import Environment
from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""
rew_buffer = deque([0.0], maxlen=100)
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model.to(device)
#data = data.to(device)
writer = SummaryWriter()

EPISODES = 10000
ALPHA = 1.5e-4
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
EPSILON = 1
EPSILON_END = 0.025
FINAL_EXP_FRAME = 100000
TARGET_UPDATE_FREQUENCY = 10000
SAVE_MODEL_AFTER = 5000
DECAY_EPSILON_AFTER = 10000
LEARNING_RATE = 1.5e-4
class Agent_DQN(Agent):
    def __init__(self, env: Environment, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """
        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.maxlen = 10000
        self.replay_buff = deque([],maxlen = self.maxlen)
        self.Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
        self.training_steps = 0
        self.env = env
        in_channel = 4

        self.op_action = self.env.action_space.n
        self.target_Q_net = DQN(in_channel,self.op_action).to(device)
        self.Q_Net = DQN(in_channel,self.op_action).to(device)
        self.target_Q_net.load_state_dict(self.Q_Net.state_dict())
        self.optimizer = optim.Adam(self.Q_Net.parameters(),lr=LEARNING_RATE)
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #


        
        ###########################
        pass
    
    
    def make_action(self, observation: np.ndarray, test: bool =True) -> int:
        """
        Returns predicted action of your agent from trained model
        """
        # Get observation in correct format for network
        state = self.format_state(observation)

        # Get Q from network/model
        Q = self.Q_Net.forward(state)

        # Greedy/deterministic action
        max_q_index = torch.argmax(Q, dim=1)[0]

        return max_q_index.detach().item()
    
    def push(self,*args):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        self.replay_buff.append(self.Transition(*args))


        
        ###########################

    def get_epsilon_greedy_action(self, greedy_action : int, epsilon: float):
        probability = np.ones(self.op_action)* epsilon / self.op_action
        probability[greedy_action] += 1 - epsilon
        return np.random.choice(np.arange(self.op_action),p=probability)

    

        
        
    def replay_buffer(self,batch_size):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        obs = env.reset()
        for _ in range(MIN_REPLAY_SIZE):
            action = env.action_space.sample()
            new_obs, rew, done,info, _ = env.step(action)
            transition = (obs, action, rew, done, new_obs)
            replay_buffer.append(transition)
            obs = new_obs
            if done:
                obs = env.reset()


        
        
        ###########################
        return replay_buffer


    def train(self, no_of_episodes: int = EPISODES):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        for epi_num in range(no_of_episodes):
            print("episode:",epi_num)
            episode_reward = 0
            curr_state = self.env.reset()
            #print('cur_state',np.shape(curr_state))


            for step in count():
                if epi_num > DECAY_EPSILON_AFTER:
                    epsilon = np.interp(step, [0, FINAL_EXPL_FRAME], [EPSILON, EPSILON_END])
                else:
                    epsilon = EPSILON
                action = self.get_epsilon_greedy_action(self.make_action(curr_state), epsilon)
                next_state, reward, done,info,_ = self.env.step(action)

                # Convert numpy arrays/int to tensors
                curr_state_t = self.format_state(curr_state)
                next_state_t = self.format_state(next_state)
                action_t = torch.tensor([action], device=device)
                reward_t = torch.tensor([reward], device=device)

                self.push(curr_state_t, action_t, reward_t, next_state_t)

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
                self.target_Q_net.load_state_dict(self.Q_Net.state_dict())

            if epi_num % SAVE_MODEL_AFTER == 0:
                torch.save(self.Q_Net.state_dict(), "vanilla_dqn_model.pth")

        torch.save(self.Q_Net.state_dict(), "vanilla_dqn_model.pth")
        print("Complete")
        writer.flush()
        writer.close()

    def train1(self, no_of_episodes: int = EPISODES):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        for epi_num in range(no_of_episodes):
            print("episode:",epi_num)
            episode_reward = 0
            curr_state = self.env.reset()
            #print('cur_state',np.shape(curr_state))


            for step in count():
                if epi_num > DECAY_EPSILON_AFTER:
                    epsilon = np.interp(step, [0, FINAL_EXPL_FRAME], [EPSILON, EPSILON_END])
                else:
                    epsilon = EPSILON
                action = self.get_epsilon_greedy_action(self.make_action(curr_state), epsilon)
                next_state, reward, done,info,_ = self.env.step(action)

                # Convert numpy arrays/int to tensors
                curr_state_t = self.format_state(curr_state)
                next_state_t = self.format_state(next_state)
                action_t = torch.tensor([action], device=device)
                reward_t = torch.tensor([reward], device=device)

                self.push(curr_state_t, action_t, reward_t, next_state_t)

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
                self.target_Q_net.load_state_dict(self.Q_Net.state_dict())

            if epi_num % SAVE_MODEL_AFTER == 0:
                torch.save(self.Q_Net.state_dict(), "vanilla_dqn_model.pth")

        torch.save(self.Q_Net.state_dict(), "vanilla_dqn_model.pth")
        print("Complete")
        writer.flush()
        writer.close()


    def optimize_model(self) -> None:
        """
        """
        if len(self.replay_buff) < BUFFER_SIZE:
            return

        self.training_steps += 1

        transitions = self.replay_buffer(BATCH_SIZE)

        # Convert batch array of transitions to Transition of batch arrays
        batch = self.Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_terminal_next_state_batch = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        sav_t = self.Q_Net(state_batch)
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
        for param in self.Q_Net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        ###########################
    @staticmethod
    def format_state(state: np.ndarray) -> torch.Tensor:
        """
        """
        state = np.asarray(state, dtype=np.float32) / 255

        # Transpose into torch order (CHW)
        state = state.transpose(2, 0, 1)

        # Add a batch dimension (BCHW)
        return torch.from_numpy(state).unsqueeze(0)
