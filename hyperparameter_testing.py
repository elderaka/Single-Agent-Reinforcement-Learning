"""
Hyperparameter testing module for Frozen Lake reinforcement learning project.
Implements grid search and visualization of different hyperparameter combinations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import os
from typing import List, Dict, Tuple
import json
from datetime import datetime

from config import *
from agent import Agent
from env import FrozenLakeEnv
from utils import train_agent

class HyperparameterTester:
    def __init__(self):
        self.results = []
        self.best_params = None
        self.best_score = float('-inf')
        
    def create_parameter_grid(self) -> List[Dict]:
        """Create a grid of hyperparameter combinations to test."""
        learning_rates = [0.01, 0.05, 0.1, 0.2]
        discount_factors = [0.9, 0.95, 0.99]
        exploration_rates = [0.1, 0.2, 0.3]
        
        param_grid = []
        for lr, gamma, epsilon in product(learning_rates, discount_factors, exploration_rates):
            param_grid.append({
                'alpha': lr,
                'gamma': gamma,
                'epsilon': epsilon
            })
        return param_grid
    
    def run_grid_search(self, grid_size: int = 4, episodes: int = 1000):
        """Run grid search over hyperparameter combinations."""
        param_grid = self.create_parameter_grid()
        
        for params in param_grid:
            print(f"\nTesting parameters: {params}")
            
            # Create environment and agent with current parameters
            env = FrozenLakeEnv(grid_size=grid_size)
            agent = Agent(
                state_size=env.observation_space.n,
                action_size=env.action_space.n,
                alpha=params['alpha'],
                gamma=params['gamma'],
                epsilon=params['epsilon']
            )
            
            # Train agent
            rewards, steps, success_rate = train_agent(
                env, agent, episodes=episodes,
                min_episodes=MIN_EPISODES,
                patience=PATIENCE,
                target_success_rate=TARGET_SUCCESS_RATE
            )
            
            # Calculate average metrics
            avg_reward = np.mean(rewards)
            avg_steps = np.mean(steps)
            
            # Store results
            result = {
                'parameters': params,
                'avg_reward': avg_reward,
                'avg_steps': avg_steps,
                'success_rate': success_rate,
                'rewards': rewards,
                'steps': steps
            }
            
            self.results.append(result)
            
            # Update best parameters
            if avg_reward > self.best_score:
                self.best_score = avg_reward
                self.best_params = params
                
            print(f"Results - Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}")
    
    def visualize_results(self):
        """Visualize the results of the grid search."""
        if not self.results:
            print("No results to visualize!")
            return
            
        # Create results directory if it doesn't exist
        os.makedirs('hyperparameter_results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to DataFrame for easier analysis
        df = pd.DataFrame([
            {
                'alpha': r['parameters']['alpha'],
                'gamma': r['parameters']['gamma'],
                'epsilon': r['parameters']['epsilon'],
                'avg_reward': r['avg_reward'],
                'success_rate': r['success_rate']
            }
            for r in self.results
        ])
        
        # Create heatmap for each parameter pair
        for param1, param2 in [('alpha', 'gamma'), ('alpha', 'epsilon'), ('gamma', 'epsilon')]:
            plt.figure(figsize=(10, 8))
            pivot = df.pivot_table(
                values='avg_reward',
                index=param1,
                columns=param2
            )
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGnBu')
            plt.title(f'Average Reward by {param1} and {param2}')
            plt.savefig(f'hyperparameter_results/heatmap_{param1}_{param2}_{timestamp}.png')
            plt.close()
        
        # Save results to CSV
        df.to_csv(f'hyperparameter_results/results_{timestamp}.csv', index=False)
        
        # Save best parameters
        with open(f'hyperparameter_results/best_params_{timestamp}.json', 'w') as f:
            json.dump(self.best_params, f, indent=4)
            
        print(f"\nBest parameters found: {self.best_params}")
        print(f"Best average reward: {self.best_score:.2f}")

def main():
    tester = HyperparameterTester()
    tester.run_grid_search()
    tester.visualize_results()

if __name__ == "__main__":
    main() 