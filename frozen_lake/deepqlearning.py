import gymnasium as gym
import numpy as np
import random
import torch
from torch import nn
from math import sqrt, pi
from replay_memory import ReplayMemory
from model import DeepQLearningNetwork

class DeepQLearning():
    BATCH_SIZE = 32 
    BUFFER_SIZE = 10000
    LR = 0.005
    DISCOUNT = 0.95 
    SYNC_STEPS = 10 
    loss_function = nn.MSELoss() 
    optimizer = None
    actions = ["Left", "Down", "Right", "Up"]

    def train(self, episodes, is_slippery=False):
        
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery, render_mode="human")
        number_of_states = env.observation_space.n
        number_of_actions = env.action_space.n
        
        EPS = 1 
        memory = ReplayMemory(self.BUFFER_SIZE)

        policy_dqn = DeepQLearningNetwork(number_of_states * 2, number_of_states, number_of_actions)
        target_dqn = DeepQLearningNetwork(number_of_states * 2, number_of_states, number_of_actions)

        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.SGD(policy_dqn.parameters(), lr=self.LR)

        global returns_per_episode
        returns_per_episode = np.zeros(episodes)
        successful_episodes = 0

        epsilon_history = []

        step_count=0
            
        for i in range(episodes):
            state = env.reset()[0]
            goal = np.random.randint(0, number_of_states)
            terminated = False     
            truncated = False
            episode_return = 0
            episode_steps = 0

            while(not terminated and not truncated):

                if random.random() < EPS:
                    action = env.action_space.sample() 
                else:
                    with torch.no_grad():
                        state_goal = self.state_goal_tensor(state, goal, number_of_states)
                        action = policy_dqn(state_goal).argmax().item()

                new_state, reward, terminated, truncated, _ = env.step(action)
                memory.append((state, action, new_state, reward, terminated, goal))
                state = new_state
                episode_return += reward
                step_count += 1
                episode_steps += 1
                if reward == 1:
                    print(f"Episode {i + 1}, Step {episode_steps}: Reward equals 1")

            returns_per_episode[i] = episode_return
            if episode_return > 0:
                successful_episodes += 1

            if len(memory) > self.BATCH_SIZE:
                mini_batch = memory.HER(self.BATCH_SIZE)
                self.optimize(mini_batch, policy_dqn, target_dqn)   

                EPS = max(EPS - 1 / episodes, 0)
                epsilon_history.append(EPS)

                if step_count > self.SYNC_STEPS:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        env.close()

        torch.save(policy_dqn.state_dict(), "best_model.pt")

        sum_returns = np.zeros(episodes)

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        number_of_states = policy_dqn.first_layer.in_features // 2

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated, goal in mini_batch:

            if terminated: 
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.DISCOUNT * target_dqn(self.state_goal_tensor(new_state, goal, number_of_states)).max())

            current_q = policy_dqn(self.state_goal_tensor(state, goal, number_of_states))
            current_q_list.append(current_q)

            target_q = target_dqn(self.state_goal_tensor(state, goal, number_of_states)) 
            target_q[action] = target
            target_q_list.append(target_q)

        mse_loss = self.loss_function(torch.stack(current_q_list), torch.stack(target_q_list))
        mae_loss = sqrt(2/pi) * sqrt(mse_loss)
        
        global accuracy
        accuracy = (1-mae_loss) * 100
        print(f"Accuracy: %{accuracy:.2f}")

        self.optimizer.zero_grad()
        mse_loss.backward()
        self.optimizer.step()

    def state_goal_tensor(self, state:int, goal:int, number_of_states:int):
        input_tensor = torch.zeros(number_of_states * 2)
        input_tensor[state] = 1
        input_tensor[number_of_states + goal] = 1
        return input_tensor

    def test(self, episodes, is_slippery=False):
        
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery, render_mode="rgb_array")
        number_of_states = env.observation_space.n
        number_of_actions = env.action_space.n

        policy_dqn = DeepQLearningNetwork(number_of_states * 2, number_of_states, number_of_actions) 
        policy_dqn.load_state_dict(torch.load("best_model.pt"))
        policy_dqn.eval()  

        for i in range(episodes):
            state = env.reset()[0]  
            goal = np.random.randint(0, number_of_states)
            terminated = False    
            truncated = False                

            while(not terminated and not truncated):  
                with torch.no_grad():
                    action = policy_dqn(self.state_goal_tensor(state, goal, number_of_states)).argmax().item()

                state, reward, terminated, truncated, _ = env.step(action)

        env.close()