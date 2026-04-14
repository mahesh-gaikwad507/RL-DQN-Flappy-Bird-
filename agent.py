import gymnasium as gym
import flappy_bird_gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import yaml
import os
import itertools
from dqn import DQN
from experience_replay import ReplayMemory

# Check for Mac GPU (MPS), Nvidia (CUDA), or CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, hyperparameter_set):
        with open('parameters.yaml', 'r') as f:
            all_params = yaml.safe_load(f)
            params = all_params[hyperparameter_set]

        self.env_id = params['env_id']
        self.epsilon_init = params['epsilon_init']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.replay_memory_size = params['replay_memory_size']
        self.mini_batch_size = params['mini_batch_size']
        self.network_sync_rate = params['network_sync_rate']
        self.alpha = params['alpha']
        self.gamma = params['gamma']
        self.reward_threshold = params['reward_threshold']

        self.MODEL_FILE = f"runs/{hyperparameter_set}.pt"
        if not os.path.exists('runs'):
            os.makedirs('runs')

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

    def run(self, is_training=True, render=False):
        # Initialize Environment
        render_mode = "human" if render else None
        env = gym.make(self.env_id, render_mode=render_mode)
        
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # Initialize Networks
        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            steps_done = 0
            best_reward = float("-inf")
        else:
            # Load trained model for testing
            if os.path.exists(self.MODEL_FILE):
                policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()
            epsilon = 0 # No randomness during testing

        # Training/Testing Loop
        for episode in itertools.count():
            state, _ = env.reset()
            episode_reward = 0
            terminated = False

            while not terminated:
                # Epsilon-greedy action selection
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    state_t = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action = policy_dqn(state_t).argmax().item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                
                # --- TURBO REWARD SHAPING ---
                if terminated:
                    reward = -20.0  # Heavy penalty for any crash
                elif reward > 0:
                    reward = 25.0   # Massive reward for passing a pipe (Game Changer)
                else:
                    reward = 0.2    # Survival incentive
                
                done = terminated or truncated
                episode_reward += reward

                if is_training:
                    memory.append((state, action, next_state, reward, done))
                    steps_done += 1
                    
                    # Learn if memory is sufficient
                    if len(memory) > self.mini_batch_size:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)
                    
                    # Sync Target Network
                    if steps_done % self.network_sync_rate == 0:
                        target_dqn.load_state_dict(policy_dqn.state_dict())

                state = next_state
                if done:
                    break

            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                
                # Save the model if it's the best one yet
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    print(f"--> NEW BEST: {episode_reward:.1f} (Model Saved)")
            
            # Print status every 10 episodes
            if (episode + 1) % 10 == 0 or not is_training:
                print(f"Episode: {episode+1} | Reward: {episode_reward:.1f} | Epsilon: {epsilon:.4f}")

            # Exit logic for testing
            if not is_training and episode >= 5:
                break

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, dones = zip(*mini_batch)

        states = torch.tensor(np.array(states), dtype=torch.float, device=device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float, device=device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float, device=device)
        dones = torch.tensor(np.array(dones), dtype=torch.bool, device=device)

        # Current Q-values
        current_q_values = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values
        with torch.no_grad():
            max_next_q_values = target_dqn(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values * (~dones))

        # Optimize
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train or test RL agent.')
    parser.add_argument('hyperparameters', help='Set from parameters.yaml')
    parser.add_argument('--train', action='store_true', help='Train the model')
    args = parser.parse_args()

    my_agent = Agent(args.hyperparameters)
    my_agent.run(is_training=args.train, render=not args.train)