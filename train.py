import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from collections import deque, namedtuple
import cv2
import time
import gc
from tqdm import tqdm

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories
os.makedirs("models", exist_ok=True)

# Experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# Custom Environment Wrappers
class SkipFrame(gym.Wrapper):
    """Skip frames to speed up training"""
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
        
    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
                
        return obs, total_reward, done, info

class GrayScaleResize(gym.ObservationWrapper):
    """Process frames by grayscaling and resizing"""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0, 
            shape=(84, 90), 
            dtype=np.float32
        )
        
    def observation(self, obs):
        # Convert to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize to 84x90
        resized = cv2.resize(gray, (90, 84), interpolation=cv2.INTER_AREA)
        # Normalize
        normalized = resized / 255.0
        return normalized

class FrameStack(gym.Wrapper):
    """Stack frames to capture temporal information"""
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0,
            shape=(num_stack, 84, 90),
            dtype=np.float32
        )
        
    def reset(self):
        obs = self.env.reset()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_observation()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        return np.array(self.frames)

# Noisy Linear Layer
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Reset parameters
        self.reset_parameters()
        
        # Register buffers for noise
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize the parameters"""
        mu_range = 1 / np.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        """Generate factorized Gaussian noise"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        """Reset the factorized noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Outer product
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        """Forward pass with noise"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

# Dueling CNN architecture with Rainbow components
class DuelingCNN(nn.Module):
    def __init__(self, in_channels, num_actions, sigma_init=0.5):
        super(DuelingCNN, self).__init__()
        
        # Feature extraction layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 84, 90)
            feature_size = self.conv_layers(dummy_input).shape[1]
        
        # Value stream (state value V(s))
        self.value_stream = nn.Sequential(
            NoisyLinear(feature_size, 512, sigma_init),
            nn.ReLU(),
            NoisyLinear(512, 1, sigma_init)
        )
        
        # Advantage stream (action advantage A(s,a))
        self.advantage_stream = nn.Sequential(
            NoisyLinear(feature_size, 512, sigma_init),
            nn.ReLU(),
            NoisyLinear(512, num_actions, sigma_init)
        )
    
    def forward(self, x):
        """Forward pass combining value and advantage streams"""
        features = self.conv_layers(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage (dueling architecture)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    
    def reset_noise(self):
        """Reset noise for all noisy layers"""
        # Reset noise in value stream
        for module in self.value_stream.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
        
        # Reset noise in advantage stream
        for module in self.advantage_stream.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# Intrinsic Curiosity Module (ICM)
class ICMModule(nn.Module):
    def __init__(self, input_shape, action_dim=12, feature_dim=256):
        super(ICMModule, self).__init__()
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
            nn.ReLU()
        )
        
        # Forward model (predicts next state features)
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        
        # Inverse model (predicts action from state transitions)
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        
        # Store action dimension
        self.action_dim = action_dim
        
    def forward(self, state, next_state, action):
        # Encode states
        state_feat = self.encoder(state)
        next_state_feat = self.encoder(next_state)
        
        # One-hot encode action with fixed number of classes
        # Make sure action is a long tensor
        action = action.long()
        
        # Safe one-hot encoding with error checks
        try:
            # Check if action values are within valid range
            if torch.any(action >= self.action_dim) or torch.any(action < 0):
                print(f"Warning: Action values out of range: {action.min().item()} to {action.max().item()}")
                # Clip actions to valid range
                action = torch.clamp(action, 0, self.action_dim - 1)
            
            action_one_hot = F.one_hot(action, num_classes=self.action_dim).float()
        except Exception as e:
            print(f"Error in one-hot encoding: {e}")
            print(f"Action shape: {action.shape}, values: {action}")
            # Fallback: create zeros tensor and manually set indices
            action_one_hot = torch.zeros(action.shape[0], self.action_dim, device=action.device)
            for i in range(action.shape[0]):
                idx = min(max(0, action[i].item()), self.action_dim - 1)
                action_one_hot[i, idx] = 1.0
        
        # Forward dynamics: predict next state features
        forward_input = torch.cat([state_feat, action_one_hot], dim=1)
        pred_next_state_feat = self.forward_model(forward_input)
        
        # Inverse dynamics: predict action
        inverse_input = torch.cat([state_feat, next_state_feat], dim=1)
        pred_action = self.inverse_model(inverse_input)
        
        return state_feat, next_state_feat, pred_next_state_feat, pred_action

# Prioritized Experience Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha      # Priority exponent
        self.beta_start = beta_start  # Importance sampling weight
        self.beta_frames = beta_frames
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.frame_idx = 0
        
    def beta_by_frame(self, frame_idx):
        """Linearly anneal beta from beta_start to 1.0"""
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        # Get max priority for new experience
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Experience(state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = Experience(state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample batch with priorities"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
        
        # Get probabilities from priorities
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices and get experiences
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        beta = self.beta_by_frame(self.frame_idx)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        
        # Convert to torch tensors
        states = torch.FloatTensor(np.array([s.state for s in samples])).to(device)
        actions = torch.LongTensor(np.array([s.action for s in samples])).to(device)
        rewards = torch.FloatTensor(np.array([s.reward for s in samples])).to(device)
        next_states = torch.FloatTensor(np.array([s.next_state for s in samples])).to(device)
        dones = torch.FloatTensor(np.array([s.done for s in samples])).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        self.frame_idx += 1
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, priorities):
        """Update priorities after learning"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # Small constant to avoid zero priority

# Rainbow ICM Agent
class RainbowICMAgent:
    def __init__(self, state_size, action_size, 
                 lr=0.0001, gamma=0.99,
                 buffer_size=10000, batch_size=32,
                 target_update=1000, n_step=3,
                 icm_lr=0.0001, intrinsic_reward_scale=0.01, 
                 icm_beta=0.2, feature_dim=256, 
                 sigma_init=0.5):
        
        self.state_size = state_size  # (num_stack, height, width)
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.icm_beta = icm_beta  # Weight between forward and inverse loss
        self.intrinsic_reward_scale = intrinsic_reward_scale
        
        # Create networks
        self.online_net = DuelingCNN(state_size[0], action_size, sigma_init).to(device)
        self.target_net = DuelingCNN(state_size[0], action_size, sigma_init).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Create ICM network
        self.icm = ICMModule(state_size, action_size, feature_dim).to(device)
        
        # Setup optimizers
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=icm_lr)
        
        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(buffer_size)
        
        # Training tracking
        self.frame_idx = 0
        
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            q_values = self.online_net(state)
            action = q_values.max(1)[1].item()
            
        self.frame_idx += 1
        
        return action
    
    def compute_intrinsic_reward(self, state, next_state, action):
        """Compute intrinsic reward using ICM - with no device requirement"""
        try:
            # Determine device from model
            device = next(self.icm.parameters()).device
            
            # Ensure state and next_state are properly formatted for the network
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(device)
            if isinstance(next_state, np.ndarray):
                next_state = torch.FloatTensor(next_state).to(device)
                
            # Make sure they have batch dimension
            if state.dim() == 3:  # Missing batch dimension
                state = state.unsqueeze(0)
            if next_state.dim() == 3:  # Missing batch dimension
                next_state = next_state.unsqueeze(0)
                
            # Convert action to tensor and ensure proper shape
            if isinstance(action, int):
                action = torch.LongTensor([action]).to(device)
            elif isinstance(action, np.ndarray):
                action = torch.LongTensor(action).to(device)
                
            # Make sure action has the right shape
            if action.dim() == 0:  # Scalar tensor
                action = action.unsqueeze(0)
                
            # Run ICM forward pass with error handling
            with torch.no_grad():
                try:
                    state_feat, next_state_feat, pred_next_state_feat, _ = self.icm(state, next_state, action)
                    
                    # Compute prediction error (using MSE)
                    prediction_error = F.mse_loss(pred_next_state_feat, next_state_feat, reduction='none').sum(dim=1)
                    
                    # Scale the intrinsic reward
                    intrinsic_reward = self.intrinsic_reward_scale * prediction_error.item()
                except Exception as e:
                    # Fallback to a small exploration bonus
                    intrinsic_reward = 0.01
                    
            return intrinsic_reward
        except Exception as e:
            # Provide a small default intrinsic reward as fallback
            return 0.01
    
    def learn(self):
        """Update networks based on sampled batch"""
        if len(self.buffer.buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(self.batch_size)
        
        # Train ICM and get intrinsic rewards
        # Compute features and predictions
        state_feat, next_state_feat, pred_next_state_feat, pred_actions = self.icm(states, next_states, actions)
        
        # Forward model loss (prediction error)
        forward_loss = F.mse_loss(pred_next_state_feat, next_state_feat.detach())
        
        # Inverse model loss (action prediction)
        inverse_loss = F.cross_entropy(pred_actions, actions)
        
        # Total ICM loss
        icm_loss = (1 - self.icm_beta) * inverse_loss + self.icm_beta * forward_loss
        
        # Update ICM
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()
        
        # Train Rainbow DQN
        # Current Q values
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values (Double DQN)
        with torch.no_grad():
            next_q_values = self.online_net(next_states)
            best_actions = next_q_values.max(1)[1]
            next_q_values = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute TD error
        td_error = (q_values - target_q_values).detach().cpu().numpy()
        
        # Compute loss
        loss = (F.smooth_l1_loss(q_values, target_q_values, reduction='none') * weights).mean()
        
        # Update online network
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()
        
        # Update target network periodically
        if self.frame_idx % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        # Update priorities
        self.buffer.update_priorities(indices, np.abs(td_error))
        
        # Reset noise
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        
        return loss.item(), icm_loss.item()
    
    def save(self, filename='rainbow_icm_model.pth'):
        """Save model"""
        torch.save(self.online_net.state_dict(), filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename='rainbow_icm_model.pth'):
        """Load model"""
        self.online_net.load_state_dict(torch.load(filename, map_location=device))
        self.target_net.load_state_dict(self.online_net.state_dict())
        print(f"Model loaded from {filename}")

# Utility function to create Mario environment
def make_mario_env(skip_frames=4):
    """Create and wrap Mario environment"""
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip_frames)
    env = GrayScaleResize(env)
    env = FrameStack(env, 4)
    return env

# Training function
def train(num_episodes=10000, 
          lr=0.0001, 
          gamma=0.9,
          buffer_size=50000, 
          batch_size=32,
          target_update=1000, 
          icm_lr=0.0001, 
          intrinsic_reward_scale=0.01,
          icm_beta=0.2, 
          feature_dim=256,
          eval_frequency=100,
          save_frequency=100):
    
    # Create environment
    env = make_mario_env()
    
    # Get state and action sizes
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    
    # Create agent
    agent = RainbowICMAgent(
        state_size=state_size,
        action_size=action_size,
        lr=lr,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update=target_update,
        icm_lr=icm_lr,
        intrinsic_reward_scale=intrinsic_reward_scale,
        icm_beta=icm_beta,
        feature_dim=feature_dim
    )
    
    # Try to load existing model
    try:
        agent.load()
    except:
        print("No existing model found. Training from scratch.")
    
    # For tracking progress
    best_reward = 0
    rewards = []
    
    # Main training loop
    for episode in range(1, num_episodes + 1):
        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0
        intrinsic_reward_total = 0
        steps = 0
        losses = []
        icm_losses = []
        
        # Episode start time
        episode_start = time.time()
        
        # Run episode
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Get intrinsic reward
            intrinsic_reward = agent.compute_intrinsic_reward(state, next_state, action)
            total_reward_with_intrinsic = reward + intrinsic_reward
            
            # Store experience
            agent.buffer.add(state, action, total_reward_with_intrinsic, next_state, done)
            
            # Learn
            if steps % 4 == 0:  # Learn every 4 steps
                learn_output = agent.learn()
                if learn_output:
                    loss, icm_loss = learn_output
                    losses.append(loss)
                    icm_losses.append(icm_loss)
            
            # Update state and rewards
            state = next_state
            total_reward += reward
            intrinsic_reward_total += intrinsic_reward
            steps += 1
            
            # Garbage collection every 100 steps
            if steps % 100 == 0:
                gc.collect()
        
        # Episode duration
        episode_duration = time.time() - episode_start
        
        # Calculate average losses
        avg_loss = np.mean(losses) if losses else 0
        avg_icm_loss = np.mean(icm_losses) if icm_losses else 0
        
        # Store episode reward
        rewards.append(total_reward)
        
        # Print episode information
        print(f"Episode {episode} - "
              f"Steps: {steps}, "
              f"Reward: {total_reward:.2f}, "
              f"Intrinsic: {intrinsic_reward_total:.2f}, "
              f"DQN Loss: {avg_loss:.4f}, "
              f"ICM Loss: {avg_icm_loss:.4f}, "
              f"Duration: {episode_duration:.2f}s")
        
        # Save if best episode
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save('models/rainbow_icm_best.pth')
            print(f"New best reward: {best_reward:.2f}")
        
        # Periodic save
        if episode % save_frequency == 0:
            agent.save(f'models/rainbow_icm_episode_{episode}.pth')
            agent.save()  # Save latest model
            
        # Periodic evaluation
        if episode % eval_frequency == 0:
            evaluate(agent, 5)
    
    # Final save
    agent.save('models/rainbow_icm_final.pth')
    agent.save()
    
    # Close environment
    env.close()
    
    return rewards

# Evaluation function
def evaluate(agent, num_episodes=10):
    """Evaluate agent performance"""
    env = make_mario_env()
    rewards = []
    
    print("\nEvaluating agent...")
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            steps += 1
        
        rewards.append(total_reward)
        print(f"Eval Episode {episode}: Reward {total_reward:.2f}, Steps {steps}")
    
    avg_reward = np.mean(rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
    
    env.close()
    return avg_reward

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    # Train agent
    train(
        num_episodes=10000,
        lr=0.0001,
        gamma=0.9,
        buffer_size=50000,
        batch_size=32,
        target_update=1000,
        icm_lr=0.0001,
        intrinsic_reward_scale=0.01,
        icm_beta=0.2,
        feature_dim=256,
        eval_frequency=100,
        save_frequency=100
    )