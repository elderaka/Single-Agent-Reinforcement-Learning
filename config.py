"""
Configuration file for the Frozen Lake reinforcement learning project.
Contains hyperparameters and settings for the environment and agent.
"""

# Environment settings
GRID_SIZES = [4, 5, 8]  # Different grid sizes to test
START_POS = (0, 0)      # Default start position

# Agent hyperparameters
ALPHA = 0.1     # Learning rate
GAMMA = 0.99    # Discount factor
EPSILON = 0.2   # Exploration rate (increased for more exploration)

# Training settings
EPISODES = 5000  # Increased number of episodes for better learning
METHODS = ["sarsa", "q_learning"]  # Learning methods to compare

# File paths
SAVE_DIR = "saved_models"
Q_TABLE_PREFIX = "q_table_"
POLICY_PREFIX = "policy_"

# Visualization settings
PLOT_REWARDS = True
SAVE_PLOTS = True
PLOT_DIR = "plots" 