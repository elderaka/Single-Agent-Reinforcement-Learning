import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any, Optional
from env import GridWorld
import logging

def plot_rewards(rewards: List[float], title: str, save_path: Optional[str] = None) -> None:
    """
    Plot the rewards per episode.
    
    Args:
        rewards: List of rewards per episode
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot rewards
        plt.plot(rewards, label='Episode Reward')
        
        # Add moving average
        window_size = 100
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(rewards)), moving_avg, 
                    label=f'{window_size}-Episode Moving Average', color='red')
        
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.legend()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logging.error(f"Error plotting rewards: {str(e)}")
        raise

def plot_policy(policy: np.ndarray, env: GridWorld, title: str, 
                save_path: Optional[str] = None) -> None:
    """
    Plot the policy as a grid with arrows and environment features.
    
    Args:
        policy: Policy array (grid_size x grid_size)
        env: GridWorld environment instance
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    try:
        # Arrow directions (up, down, left, right)
        arrows = {
            0: '↑',
            1: '↓',
            2: '←',
            3: '→'
        }
        
        # Colors for different tile types
        colors = {
            GridWorld.FROZEN: 'lightblue',
            GridWorld.HOLE: 'black',
            GridWorld.START: 'green',
            GridWorld.GOAL: 'gold'
        }
        
        plt.figure(figsize=(8, 8))
        plt.title(title)
        
        # Create grid
        for i in range(env.size):
            for j in range(env.size):
                # Draw colored squares for tile types
                tile_type = env.map[i, j]
                plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                                facecolor=colors[tile_type],
                                                edgecolor='black'))
                
                # Add arrows for policy
                if tile_type != GridWorld.HOLE and tile_type != GridWorld.GOAL:
                    action = policy[i, j]
                    plt.text(j, i, arrows[action], ha='center', va='center', 
                            fontsize=20, color='black')
        
        plt.xlim(-0.5, env.size-0.5)
        plt.ylim(-0.5, env.size-0.5)
        plt.grid(True)
        plt.gca().invert_yaxis()  # Invert y-axis to match matrix indexing
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=colors[GridWorld.FROZEN], label='Frozen'),
            plt.Rectangle((0, 0), 1, 1, facecolor=colors[GridWorld.HOLE], label='Hole'),
            plt.Rectangle((0, 0), 1, 1, facecolor=colors[GridWorld.START], label='Start'),
            plt.Rectangle((0, 0), 1, 1, facecolor=colors[GridWorld.GOAL], label='Goal')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logging.error(f"Error plotting policy: {str(e)}")
        raise

def save_results(results: Dict[str, Any], filename: str) -> None:
    """
    Save training results to a file.
    
    Args:
        results: Dictionary containing results
        filename: Path to save the results
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            for key, value in results.items():
                if isinstance(value, dict):
                    f.write(f"\n{key}:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"  {subkey}: {subvalue}\n")
                else:
                    f.write(f"{key}: {value}\n")
                    
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
        raise

def create_directories() -> None:
    """Create necessary directories for saving models and plots."""
    try:
        directories = ["saved_models", "plots", "logs"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    except Exception as e:
        logging.error(f"Error creating directories: {str(e)}")
        raise

def load_results(filename: str) -> Dict[str, Any]:
    """
    Load training results from a file.
    
    Args:
        filename: Path to the results file
        
    Returns:
        Dictionary containing the loaded results
    """
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Results file {filename} not found")
            
        results = {}
        with open(filename, 'r') as f:
            current_dict = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.endswith(':'):
                    current_dict = {}
                    results[line[:-1]] = current_dict
                else:
                    key, value = line.split(': ', 1)
                    if current_dict is not None:
                        current_dict[key] = value
                    else:
                        results[key] = value
                        
        return results
        
    except Exception as e:
        logging.error(f"Error loading results: {str(e)}")
        raise 