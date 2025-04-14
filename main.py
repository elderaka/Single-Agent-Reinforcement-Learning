import time
from typing import Dict, Any
import numpy as np

from env import GridWorld
from agent import Agent
from config import *
from utils import plot_rewards, plot_policy, save_results, create_directories

def train_agent(grid_size: int, method: str) -> Dict[str, Any]:
    """
    Train an agent on the Frozen Lake environment.
    
    Args:
        grid_size: Size of the grid
        method: Learning method ("sarsa" or "q_learning")
        
    Returns:
        Dictionary containing training results
    """
    # Create environment and agent
    env = GridWorld(size=grid_size, is_slippery=True)
    agent = Agent(
        state_space_size=env.get_state_space_size(),
        action_space_size=env.get_action_space_size(),
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        method=method
    )
    
    # Training metrics
    episode_rewards = []
    episode_steps = []
    success_rate = 0
    start_time = time.time()
    
    # Training loop
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        # Choose initial action
        action = agent.choose_action(state, grid_size)
        
        while not done:
            # Take action and observe next state and reward
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Choose next action
            next_action = agent.choose_action(next_state, grid_size)
            
            # Update Q-values
            agent.update(state, action, reward, next_state, next_action, done, grid_size)
            
            # Move to next state
            state = next_state
            action = next_action
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # Track success rate (reaching goal)
        if reward > 0:  # Positive reward means reached goal
            success_rate += 1
    
    # Calculate training time and success rate
    training_time = time.time() - start_time
    success_rate = success_rate / EPISODES
    
    # Get final policy and action counts
    policy = agent.get_policy(grid_size)
    action_counts = agent.get_action_counts()
    
    # Save Q-table
    q_table_path = f"{SAVE_DIR}/{Q_TABLE_PREFIX}{method}_{grid_size}x{grid_size}.pkl"
    agent.save_q_table(q_table_path)
    
    # Plot results if enabled
    if PLOT_REWARDS:
        plot_rewards(
            episode_rewards,
            f"Rewards per Episode ({method}, {grid_size}x{grid_size})",
            f"{PLOT_DIR}/rewards_{method}_{grid_size}x{grid_size}.png"
        )
    
    if SAVE_PLOTS:
        plot_policy(
            policy,
            env,
            f"Learned Policy ({method}, {grid_size}x{grid_size})",
            f"{PLOT_DIR}/policy_{method}_{grid_size}x{grid_size}.png"
        )
    
    # Return results
    return {
        "method": method,
        "grid_size": grid_size,
        "training_time": training_time,
        "success_rate": success_rate,
        "average_steps": np.mean(episode_steps),
        "final_reward": episode_rewards[-1],
        "average_reward": np.mean(episode_rewards[-100:]),  # Last 100 episodes
        "action_counts": action_counts
    }

def main():
    """Main training loop for different grid sizes and methods."""
    # Create necessary directories
    create_directories()
    
    # Train on different grid sizes and methods
    all_results = []
    for grid_size in GRID_SIZES:
        for method in METHODS:
            print(f"\nTraining {method} on {grid_size}x{grid_size} Frozen Lake...")
            results = train_agent(grid_size, method)
            all_results.append(results)
            
            # Print results
            print(f"Training completed in {results['training_time']:.2f} seconds")
            print(f"Success rate: {results['success_rate']:.2%}")
            print(f"Average steps per episode: {results['average_steps']:.2f}")
            print(f"Final reward: {results['final_reward']:.2f}")
            print(f"Average reward (last 100 episodes): {results['average_reward']:.2f}")
            print("Action counts:", results['action_counts'])
    
    # Save all results
    save_results(
        {"results": all_results},
        f"{SAVE_DIR}/training_results.txt"
    )

if __name__ == "__main__":
    main() 