import time
from typing import Dict, Any, List
import numpy as np
from collections import deque
import keyboard
import logging
import os
from datetime import datetime

from env import GridWorld
from agent import Agent
from config import *
from utils import plot_rewards, plot_policy, save_results, create_directories

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def analyze_training_conclusion(episode_rewards: List[float], episode_steps: List[int], 
                              success_rate: float) -> Dict[str, Any]:
    """
    Analyze the training results and provide a conclusion.
    
    Args:
        episode_rewards: List of rewards per episode
        episode_steps: List of steps per episode
        success_rate: Final success rate
        
    Returns:
        Dictionary containing analysis results
    """
    # Calculate metrics
    avg_reward = np.mean(episode_rewards[-100:])
    avg_steps = np.mean(episode_steps[-100:])
    reward_std = np.std(episode_rewards[-100:])
    steps_std = np.std(episode_steps[-100:])
    
    # Determine if training was successful
    is_successful = (
        success_rate >= TARGET_SUCCESS_RATE and
        avg_reward >= REWARD_THRESHOLD and
        avg_steps <= (GRID_SIZE * STEPS_THRESHOLD)
    )
    
    # Provide recommendations
    recommendations = []
    if not is_successful:
        if success_rate < TARGET_SUCCESS_RATE:
            recommendations.append("Try increasing epsilon for more exploration")
        if avg_reward < REWARD_THRESHOLD:
            recommendations.append("Try adjusting alpha for more effective learning")
        if avg_steps > (GRID_SIZE * STEPS_THRESHOLD):
            recommendations.append("Agent seems stuck, check environment configuration")
    
    return {
        "is_successful": is_successful,
        "success_rate": success_rate,
        "average_reward": avg_reward,
        "average_steps": avg_steps,
        "reward_std": reward_std,
        "steps_std": steps_std,
        "recommendations": recommendations
    }

def check_manual_stop() -> bool:
    """Check if manual stop was triggered."""
    return MANUAL_STOP_FLAG

def train_agent(grid_size: int, method: str) -> Dict[str, Any]:
    """
    Train an agent on the Frozen Lake environment with automatic and manual stopping.
    
    Args:
        grid_size: Size of the grid
        method: Learning method ("sarsa" or "q_learning")
        
    Returns:
        Dictionary containing training results
    """
    try:
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
        
        # Early stopping variables
        best_success_rate = 0
        patience_counter = 0
        early_stopped = False
        manually_stopped = False
        
        # Create a deque to track recent rewards for early stopping
        recent_rewards = deque(maxlen=100)
        
        logging.info(f"\nStarting training {method} on {grid_size}x{grid_size} grid...")
        logging.info("Press 's' to stop training and view results...")
        
        # Training loop
        for episode in range(EPISODES):
            # Check for manual stop
            if keyboard.is_pressed('s'):
                manually_stopped = True
                logging.info("Training stopped manually")
                break
                
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
            
            # Update metrics
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            recent_rewards.append(total_reward)
            
            # Track success rate (reaching goal)
            if reward > 0:  # Positive reward means reached goal
                success_rate += 1
            
            # Decay epsilon
            agent.decay_epsilon()
            
            # Log progress
            if (episode + 1) % 100 == 0:
                current_success_rate = success_rate / (episode + 1)
                avg_reward = np.mean(recent_rewards)
                logging.info(
                    f"Episode {episode + 1}/{EPISODES} - "
                    f"Success Rate: {current_success_rate:.2%} - "
                    f"Avg Reward: {avg_reward:.2f} - "
                    f"Epsilon: {agent.get_epsilon():.3f}"
                )
            
            # Early stopping check
            if episode >= MIN_EPISODES:
                current_success_rate = success_rate / (episode + 1)
                
                if current_success_rate > best_success_rate:
                    best_success_rate = current_success_rate
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= PATIENCE and current_success_rate >= TARGET_SUCCESS_RATE:
                    early_stopped = True
                    logging.info("Early stopping triggered")
                    break
        
        # Calculate final metrics
        training_time = time.time() - start_time
        final_success_rate = success_rate / (episode + 1)
        
        # Get final policy and action counts
        policy = agent.get_policy(grid_size)
        action_counts = agent.get_action_counts()
        
        # Analyze training conclusion
        conclusion = analyze_training_conclusion(episode_rewards, episode_steps, final_success_rate)
        
        # Save Q-table
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        q_table_path = f"{SAVE_DIR}/{Q_TABLE_PREFIX}{method}_{grid_size}x{grid_size}_{timestamp}.pkl"
        agent.save_q_table(q_table_path)
        
        # Plot results if enabled
        if PLOT_REWARDS:
            plot_rewards(
                episode_rewards,
                f"Reward per Episode ({method}, {grid_size}x{grid_size})",
                f"{PLOT_DIR}/rewards_{method}_{grid_size}x{grid_size}_{timestamp}.png"
            )
        
        if SAVE_PLOTS:
            plot_policy(
                policy,
                env,
                f"Learned Policy ({method}, {grid_size}x{grid_size})",
                f"{PLOT_DIR}/policy_{method}_{grid_size}x{grid_size}_{timestamp}.png"
            )
        
        # Return results
        return {
            "method": method,
            "grid_size": grid_size,
            "training_time": training_time,
            "episodes_completed": episode + 1,
            "early_stopped": early_stopped,
            "manually_stopped": manually_stopped,
            "success_rate": final_success_rate,
            "average_steps": np.mean(episode_steps),
            "final_reward": episode_rewards[-1],
            "average_reward": np.mean(episode_rewards[-100:]),
            "action_counts": action_counts,
            "conclusion": conclusion
        }
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

def main():
    """Main training loop for different grid sizes and methods."""
    try:
        # Create necessary directories
        create_directories()
        
        # Train on different grid sizes and methods
        all_results = []
        for grid_size in GRID_SIZES:
            for method in METHODS:
                logging.info(f"\nTraining {method} on {grid_size}x{grid_size} grid...")
                results = train_agent(grid_size, method)
                all_results.append(results)
                
                # Log results
                logging.info("\nTraining Summary:")
                logging.info(f"Episodes completed: {results['episodes_completed']}")
                logging.info(f"Early stopped: {'Yes' if results['early_stopped'] else 'No'}")
                logging.info(f"Manually stopped: {'Yes' if results['manually_stopped'] else 'No'}")
                logging.info(f"Training time: {results['training_time']:.2f} seconds")
                logging.info(f"Success rate: {results['success_rate']:.2%}")
                logging.info(f"Average steps per episode: {results['average_steps']:.2f}")
                logging.info(f"Final reward: {results['final_reward']:.2f}")
                logging.info(f"Average reward (last 100 episodes): {results['average_reward']:.2f}")
                logging.info(f"Action counts: {results['action_counts']}")
                
                # Log conclusion
                logging.info("\nTraining Conclusion:")
                conclusion = results['conclusion']
                logging.info(f"Training successful: {'Yes' if conclusion['is_successful'] else 'No'}")
                logging.info(f"Success rate: {conclusion['success_rate']:.2%}")
                logging.info(f"Average reward: {conclusion['average_reward']:.2f} (±{conclusion['reward_std']:.2f})")
                logging.info(f"Average steps: {conclusion['average_steps']:.2f} (±{conclusion['steps_std']:.2f})")
                
                if conclusion['recommendations']:
                    logging.info("\nRecommendations for improving performance:")
                    for rec in conclusion['recommendations']:
                        logging.info(f"- {rec}")
                
                # If manually stopped, ask if user wants to continue with next configuration
                if results['manually_stopped']:
                    response = input("\nContinue to next configuration? (y/n): ")
                    if response.lower() != 'y':
                        break
        
        # Save all results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(
            {"results": all_results},
            f"{SAVE_DIR}/training_results_{timestamp}.txt"
        )
        
    except Exception as e:
        logging.error(f"Error in main loop: {str(e)}")
        raise

if __name__ == "__main__":
    main() 