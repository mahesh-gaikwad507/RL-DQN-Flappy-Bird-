import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory 
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else: 
    device = torch.device("cpu")

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def run(self, is_training=True, render=False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

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
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()

        for episode in itertools.count():
            state, _ = env.reset()
            episode_reward = 0
            terminated = False

            while not terminated and episode_reward < self.reward_threshold:
                # Epsilon-greedy action selection
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    state_t = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action = policy_dqn(state_t).argmax().item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward

                if is_training:
                    # Store as raw data to avoid constant CPU-GPU transfer
                    memory.append((state, action, next_state, reward, done))
                    steps_done += 1

                    # Optimize every step if buffer is full enough
                    if len(memory) > self.mini_batch_size:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Sync target network based on total steps
                    if steps_done % self.network_sync_rate == 0:
                        target_dqn.load_state_dict(policy_dqn.state_dict())

                state = next_state
                if done:
                    break

            # Epsilon decay happens per episode
            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    log_msg = f"Episode {episode+1}: New best reward = {episode_reward:.2f}"
                    print(log_msg)
                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

            print(f"Episode: {episode+1} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.4f}")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        # Convert entire batch to tensors at once (more efficient)
        states = torch.tensor(states, dtype=torch.float, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        with torch.no_grad():
            # Bellman Equation: Q_target = r + gamma * max(Q_target(s')) * (1 - done)
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        # Get current Q-values for the actions taken
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='Name of parameter set from yaml')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)