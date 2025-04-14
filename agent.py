import numpy as np
from typing import Tuple, Dict, Optional
import pickle
import os

class Agent:
    """
    Reinforcement Learning agent that supports both SARSA and Q-Learning algorithms.
    Uses epsilon-greedy exploration strategy.
    """
    
    def __init__(self, state_space_size: int, action_space_size: int,
                 alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1,
                 method: str = "q_learning"):
        """
        Initialize the agent with learning parameters.
        
        Args:
            state_space_size: Number of possible states
            action_space_size: Number of possible actions
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate for epsilon-greedy
            method: Learning method ("sarsa" or "q_learning")
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.method = method.lower()
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_space_size, action_space_size))
        
        # Action counts for statistics
        self.action_counts = np.zeros(action_space_size)
        
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
            action = np.argmax(self.q_table[state_idx])
        
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
    
    def save_q_table(self, filename: str) -> None:
        """Save the Q-table to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load_q_table(self, filename: str) -> None:
        """Load the Q-table from a file."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
    
    def get_policy(self, grid_size: int) -> np.ndarray:
        """
        Get the optimal policy (best action for each state).
        
        Returns:
            policy: Array of best actions for each state
        """
        policy = np.zeros(self.state_space_size, dtype=int)
        for state_idx in range(self.state_space_size):
            policy[state_idx] = np.argmax(self.q_table[state_idx])
        return policy.reshape(grid_size, grid_size)
    
    def get_action_counts(self) -> Dict[str, int]:
        """Get the count of each action taken."""
        action_names = {0: "up", 1: "down", 2: "left", 3: "right"}
        return {action_names[i]: int(count) for i, count in enumerate(self.action_counts)} 