"""
Configuration file for the Frozen Lake reinforcement learning project.
Contains hyperparameters and settings for the environment and agent.
"""

from typing import List, Tuple, Dict, Any
import os

# Environment settings
GRID_SIZES: List[int] = [4, 5, 8]  # Grid sizes to test
START_POS: Tuple[int, int] = (0, 0)  # Starting position
MAX_HOLES_RATIO: float = 0.25  # Maximum ratio of holes to grid size

# Agent hyperparameters
ALPHA: float = 0.1  # Learning rate
GAMMA: float = 0.99  # Discount factor
EPSILON: float = 0.2  # Initial exploration rate
EPSILON_MIN: float = 0.01  # Minimum exploration rate
EPSILON_DECAY: float = 0.995  # Exploration rate decay

# Training settings
EPISODES: int = 5000  # Maximum number of episodes
METHODS: List[str] = ["sarsa", "q_learning"]  # Learning methods to compare

# Early stopping criteria
MIN_EPISODES: int = 1000  # Minimum episodes before considering early stopping
PATIENCE: int = 100  # Number of episodes to wait for improvement
TARGET_SUCCESS_RATE: float = 0.8  # Target success rate for early stopping
REWARD_THRESHOLD: float = 0.9  # Target average reward threshold
STEPS_THRESHOLD: float = 2.0  # Maximum allowed average steps multiplier

# Path settings
SAVE_DIR: str = "saved_models"
Q_TABLE_PREFIX: str = "q_table_"
POLICY_PREFIX: str = "policy_"
PLOT_DIR: str = "plots"
LOG_DIR: str = "logs"

# Visualization settings
PLOT_REWARDS: bool = True
SAVE_PLOTS: bool = True
PLOT_INTERVAL: int = 100  # Interval for plotting progress

# Validation
def validate_config() -> None:
    """Validate configuration values."""
    # Validate grid sizes
    for size in GRID_SIZES:
        if size < 2:
            raise ValueError("Grid size must be at least 2x2")
    
    # Validate positions
    for size in GRID_SIZES:
        if not (0 <= START_POS[0] < size and 0 <= START_POS[1] < size):
            raise ValueError(f"Start position {START_POS} is invalid for grid size {size}")
    
    # Validate hyperparameters
    if not (0 < ALPHA <= 1):
        raise ValueError("Learning rate must be between 0 and 1")
    if not (0 < GAMMA <= 1):
        raise ValueError("Discount factor must be between 0 and 1")
    if not (0 <= EPSILON <= 1):
        raise ValueError("Exploration rate must be between 0 and 1")
    if not (0 <= EPSILON_MIN <= EPSILON):
        raise ValueError("Minimum exploration rate must be between 0 and initial epsilon")
    if not (0 < EPSILON_DECAY < 1):
        raise ValueError("Exploration decay must be between 0 and 1")
    
    # Validate training settings
    if EPISODES < MIN_EPISODES:
        raise ValueError("Total episodes must be greater than minimum episodes")
    if PATIENCE < 1:
        raise ValueError("Patience must be at least 1")
    if not (0 <= TARGET_SUCCESS_RATE <= 1):
        raise ValueError("Target success rate must be between 0 and 1")
    if not (0 <= REWARD_THRESHOLD <= 1):
        raise ValueError("Reward threshold must be between 0 and 1")
    if STEPS_THRESHOLD < 1:
        raise ValueError("Steps threshold must be at least 1")
    
    # Validate methods
    valid_methods = ["sarsa", "q_learning"]
    for method in METHODS:
        if method not in valid_methods:
            raise ValueError(f"Invalid method: {method}. Must be one of {valid_methods}")

# Create necessary directories
def create_directories() -> None:
    """Create necessary directories for saving models, plots, and logs."""
    directories = [SAVE_DIR, PLOT_DIR, LOG_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Validate configuration on import
validate_config()
create_directories() 