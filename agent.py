import numpy as np
from typing import Tuple, Dict, Optional
import pickle
import os

class Agent:
    """
    Reinforcement Learning agent that supports both SARSA and Q-Learning algorithms.
    Uses epsilon-greedy exploration strategy with decay.
    """
    
    def __init__(self, state_space_size: int, action_space_size: int,
                 alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1,
                 epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                 method: str = "q_learning"):
        """
        Initialize the agent with learning parameters.
        
        Args:
            state_space_size: Number of possible states
            action_space_size: Number of possible actions
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate at which epsilon decays
            method: Learning method ("sarsa" or "q_learning")
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.method = method.lower()
        
        # Initialize Q-table with small random values to break symmetry
        self.q_table = np.random.uniform(
            low=-0.1, high=0.1,
            size=(state_space_size, action_space_size)
        )
        
        # Action counts for statistics
        self.action_counts = np.zeros(action_space_size)
        
        # Track training progress
        self.episode_count = 0
        
    def get_state_index(self, state: Tuple[int, int], grid_size: int) -> int:
        """Convert 2D grid position to 1D state index."""
        return state[0] * grid_size + state[1]
    
    def choose_action(self, state: Tuple[int, int], grid_size: int) -> int:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state (row, col)
            grid_size: Size of the grid
            
        Returns:
            action: Chosen action index
        """
        state_idx = self.get_state_index(state, grid_size)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_space_size)
        else:
            # Break ties randomly
            max_q = np.max(self.q_table[state_idx])
            best_actions = np.where(self.q_table[state_idx] == max_q)[0]
            action = np.random.choice(best_actions)
        
        # Update action counts
        self.action_counts[action] += 1
        return action
    
    def update(self, state: Tuple[int, int], action: int, reward: float,
               next_state: Tuple[int, int], next_action: Optional[int] = None,
               done: bool = False, grid_size: int = 4) -> None:
        """
        Update Q-values using either SARSA or Q-Learning.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (required for SARSA)
            done: Whether episode is complete
            grid_size: Size of the grid
        """
        state_idx = self.get_state_index(state, grid_size)
        next_state_idx = self.get_state_index(next_state, grid_size)
        
        if self.method == "sarsa":
            if next_action is None:
                raise ValueError("next_action is required for SARSA")
            
            # SARSA update
            next_q = self.q_table[next_state_idx, next_action]
            target = reward + self.gamma * next_q * (1 - done)
            
        else:  # Q-Learning
            # Q-Learning update
            next_q = np.max(self.q_table[next_state_idx])
            target = reward + self.gamma * next_q * (1 - done)
        
        # Update Q-value
        self.q_table[state_idx, action] += self.alpha * (target - self.q_table[state_idx, action])
    
    def decay_epsilon(self) -> None:
        """Decay the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_q_table(self, filename: str) -> None:
        """Save the Q-table to a file."""
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'episode_count': self.episode_count
            }, f)
    
    def load_q_table(self, filename: str) -> None:
        """Load the Q-table from a file."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data.get('epsilon', self.epsilon)
                self.episode_count = data.get('episode_count', 0)
    
    def get_policy(self, grid_size: int) -> np.ndarray:
        """
        Get the optimal policy (best action for each state).
        
        Returns:
            policy: Array of best actions for each state
        """
        policy = np.zeros(self.state_space_size, dtype=int)
        for state_idx in range(self.state_space_size):
            # Break ties randomly
            max_q = np.max(self.q_table[state_idx])
            best_actions = np.where(self.q_table[state_idx] == max_q)[0]
            policy[state_idx] = np.random.choice(best_actions)
        return policy.reshape(grid_size, grid_size)
    
    def get_action_counts(self) -> Dict[str, int]:
        """Get the count of each action taken."""
        action_names = {0: "up", 1: "down", 2: "left", 3: "right"}
        return {action_names[i]: int(count) for i, count in enumerate(self.action_counts)}
    
    def get_q_table(self) -> np.ndarray:
        """Get the current Q-table."""
        return self.q_table.copy()
    
    def set_q_table(self, q_table: np.ndarray):
        """Set the Q-table to a new value."""
        if q_table.shape != (self.state_space_size, self.action_space_size):
            raise ValueError("Q-table shape doesn't match state and action space sizes")
        self.q_table = q_table.copy()
    
    def get_epsilon(self) -> float:
        """Get the current exploration rate."""
        return self.epsilon 