import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import time
import os
import pickle
import json

from env import GridWorld
from agent import Agent
from utils import plot_rewards, plot_policy
from config import ALPHA, GAMMA, EPSILON, EPISODES, METHODS

class ComparisonPlotWindow:
    """Window for comparing SARSA and Q-Learning training progress."""
    
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Perbandingan Training")
        self.window.geometry("1200x800")
        
        # Create figure
        self.figure = plt.figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize subplots
        self.rewards_ax = None
        self.policy_sarsa_ax = None
        self.policy_qlearning_ax = None
        
        # Close handler
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self.is_closed = False
    
    def _on_close(self):
        """Handle window close event."""
        self.is_closed = True
        plt.close(self.figure)
        self.window.destroy()
    
    def update_plot(self, sarsa_data: tuple, qlearning_data: tuple):
        """Update the comparison plots."""
        if self.is_closed:
            return
        
        # Unpack data
        sarsa_rewards, sarsa_env, sarsa_policy = sarsa_data
        qlearning_rewards, qlearning_env, qlearning_policy = qlearning_data
        
        # Clear previous plot
        self.figure.clear()
        
        # Create subplots
        gs = self.figure.add_gridspec(2, 2)
        self.rewards_ax = self.figure.add_subplot(gs[0, :])
        self.policy_sarsa_ax = self.figure.add_subplot(gs[1, 0])
        self.policy_qlearning_ax = self.figure.add_subplot(gs[1, 1])
        
        # Plot rewards
        self.rewards_ax.plot(sarsa_rewards, label='SARSA')
        self.rewards_ax.plot(qlearning_rewards, label='Q-Learning')
        self.rewards_ax.set_title("Reward per Episode")
        self.rewards_ax.set_xlabel("Episode")
        self.rewards_ax.set_ylabel("Total Reward")
        self.rewards_ax.grid(True)
        self.rewards_ax.legend()
        
        # Plot policies
        self._plot_policy(self.policy_sarsa_ax, sarsa_env, sarsa_policy, "SARSA Policy")
        self._plot_policy(self.policy_qlearning_ax, qlearning_env, qlearning_policy, "Q-Learning Policy")
        
        # Update canvas
        self.figure.tight_layout()
        self.canvas.draw()
        self.window.update()
    
    def _plot_policy(self, ax, env: GridWorld, policy: np.ndarray, title: str):
        """Helper method to plot policy grid."""
        arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        colors = {
            GridWorld.FROZEN: 'lightblue',
            GridWorld.HOLE: 'black',
            GridWorld.START: 'green',
            GridWorld.GOAL: 'gold'
        }
        
        # Draw grid cells
        for i in range(env.size):
            for j in range(env.size):
                tile_type = env.map[i, j]
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                        facecolor=colors[tile_type],
                                        edgecolor='black'))
                
                # Add policy arrows
                if tile_type != GridWorld.HOLE and tile_type != GridWorld.GOAL:
                    action = policy[i, j]
                    ax.text(j, i, arrows[action], ha='center', va='center',
                          fontsize=12, color='black')
        
        ax.set_title(title)
        ax.set_xlim(-0.5, env.size-0.5)
        ax.set_ylim(-0.5, env.size-0.5)
        ax.grid(True)
        ax.invert_yaxis()

class FrozenLakeUI:
    """UI for creating and training Frozen Lake environments."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Single Agent Reinforcement Training - Frozen Lake Environment")
        self.root.geometry("1000x800")
        
        # Environment parameters
        self.size = tk.IntVar(value=4)
        self.is_slippery = tk.BooleanVar(value=True)
        self.selected_tile = tk.StringVar(value=GridWorld.FROZEN)
        self.training_method = tk.StringVar(value=METHODS[0])
        
        # Hyperparameters
        self.alpha = tk.DoubleVar(value=ALPHA)
        self.gamma = tk.DoubleVar(value=GAMMA)
        self.epsilon = tk.DoubleVar(value=EPSILON)
        self.episodes = tk.IntVar(value=EPISODES)
        
        # Store environment and training data
        self.env: Optional[GridWorld] = None
        self.current_agent: Optional[Agent] = None
        self.grid_buttons: List[List[tk.Button]] = []
        self.start_pos: Optional[Tuple[int, int]] = None
        self.goal_pos: List[Tuple[int, int]] = []
        self.holes: List[Tuple[int, int]] = []
        
        # Threading and plot management
        self.plot_queue = queue.Queue()
        self.sarsa_thread = None
        self.qlearning_thread = None
        self.stop_training = False
        self.plot_window = None
        
        # Training statistics
        self.sarsa_stats = {'time': 0, 'counts': {}, 'rate': 0}
        self.qlearning_stats = {'time': 0, 'counts': {}, 'rate': 0}
        
        self._create_ui()
    
    def _create_ui(self):
        """Create the main UI components."""
        # Left panel for grid controls
        left_panel = ttk.Frame(self.root, padding="10")
        left_panel.grid(row=0, column=0, sticky="nsew")
        
        # Grid size controls
        ttk.Label(left_panel, text="Ukuran Grid:").grid(row=0, column=0, pady=5)
        size_entry = ttk.Entry(left_panel, textvariable=self.size, width=5)
        size_entry.grid(row=0, column=1, pady=5)
        ttk.Button(left_panel, text="Buat Grid", command=self._create_grid).grid(row=0, column=2, pady=5, padx=5)
        
        # Tile selection
        ttk.Label(left_panel, text="Pilih Tile:").grid(row=1, column=0, pady=5)
        tiles = [(GridWorld.FROZEN, "Es"), (GridWorld.HOLE, "Lubang"), 
                (GridWorld.START, "Start"), (GridWorld.GOAL, "Finish")]
        for i, (tile, name) in enumerate(tiles):
            ttk.Radiobutton(left_panel, text=name, value=tile, 
                          variable=self.selected_tile).grid(row=1, column=i+1, pady=5)
        
        # Environment options
        ttk.Checkbutton(left_panel, text="Licin?", 
                       variable=self.is_slippery).grid(row=2, column=0, pady=5)
        
        # Hyperparameter controls
        hyper_frame = ttk.LabelFrame(left_panel, text="Hyperparameters", padding="5")
        hyper_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky="ew")
        
        # Learning Rate (Alpha)
        ttk.Label(hyper_frame, text="Learning Rate (α):").grid(row=0, column=0, pady=2)
        alpha_scale = ttk.Scale(hyper_frame, from_=0.01, to=1.0, variable=self.alpha, 
                              orient="horizontal", length=150)
        alpha_scale.grid(row=0, column=1, pady=2)
        ttk.Label(hyper_frame, textvariable=self.alpha).grid(row=0, column=2, pady=2)
        
        # Discount Factor (Gamma)
        ttk.Label(hyper_frame, text="Discount Factor (γ):").grid(row=1, column=0, pady=2)
        gamma_scale = ttk.Scale(hyper_frame, from_=0.1, to=0.99, variable=self.gamma,
                              orient="horizontal", length=150)
        gamma_scale.grid(row=1, column=1, pady=2)
        ttk.Label(hyper_frame, textvariable=self.gamma).grid(row=1, column=2, pady=2)
        
        # Exploration Rate (Epsilon)
        ttk.Label(hyper_frame, text="Exploration Rate (ε):").grid(row=2, column=0, pady=2)
        epsilon_scale = ttk.Scale(hyper_frame, from_=0.01, to=1.0, variable=self.epsilon,
                                orient="horizontal", length=150)
        epsilon_scale.grid(row=2, column=1, pady=2)
        ttk.Label(hyper_frame, textvariable=self.epsilon).grid(row=2, column=2, pady=2)
        
        # Number of Episodes
        ttk.Label(hyper_frame, text="Episodes:").grid(row=3, column=0, pady=2)
        episodes_scale = ttk.Scale(hyper_frame, from_=100, to=10000, variable=self.episodes,
                                 orient="horizontal", length=150)
        episodes_scale.grid(row=3, column=1, pady=2)
        ttk.Label(hyper_frame, textvariable=self.episodes).grid(row=3, column=2, pady=2)
        
        # Training controls
        ttk.Label(left_panel, text="Training:").grid(row=4, column=0, pady=5)
        self.train_button = ttk.Button(left_panel, text="Bandingkan Metode", 
                                     command=self._toggle_training)
        self.train_button.grid(row=4, column=1, columnspan=2, pady=10)
        
        # Save/Load buttons for environment
        ttk.Button(left_panel, text="Simpan Environment", 
                  command=self._save_environment).grid(row=5, column=0, pady=5)
        ttk.Button(left_panel, text="Load Environment", 
                  command=self._load_environment).grid(row=5, column=1, pady=5)
        
        # Save/Load buttons for Q-table
        ttk.Label(left_panel, text="Q-Table:").grid(row=6, column=0, pady=5)
        ttk.Button(left_panel, text="Simpan Q-Table", 
                  command=self._save_q_table).grid(row=6, column=1, pady=5)
        ttk.Button(left_panel, text="Load Q-Table", 
                  command=self._load_q_table).grid(row=6, column=2, pady=5)
        
        # Grid display (middle panel)
        self.grid_frame = ttk.Frame(self.root, padding="10")
        self.grid_frame.grid(row=0, column=1, sticky="nsew")
        
        # Right panel for stats
        right_panel = ttk.Frame(self.root, padding="10")
        right_panel.grid(row=0, column=2, sticky="nsew")
        
        # Create two stats frames
        stats_container = ttk.Frame(right_panel)
        stats_container.pack(fill=tk.X, pady=5)
        
        self.sarsa_stats_frame = ttk.LabelFrame(stats_container, text="Statistik SARSA", padding="5")
        self.sarsa_stats_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.qlearning_stats_frame = ttk.LabelFrame(stats_container, text="Statistik Q-Learning", padding="5")
        self.qlearning_stats_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Configure grid weights
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        
        # Start the plot update loop
        self._update_plot_loop()
    
    def _update_stats(self, stats_frame: ttk.LabelFrame, training_time: float, 
                     action_counts: Dict[str, int], success_rate: float):
        """Update statistics for a specific method."""
        # Clear previous stats
        for widget in stats_frame.winfo_children():
            widget.destroy()
        
        # Training parameters
        ttk.Label(stats_frame, text=f"Learning Rate (α): {ALPHA}").pack(anchor="w")
        ttk.Label(stats_frame, text=f"Discount Factor (γ): {GAMMA}").pack(anchor="w")
        ttk.Label(stats_frame, text=f"Exploration Rate (ε): {EPSILON}").pack(anchor="w")
        
        # Training results
        ttk.Label(stats_frame, text=f"Waktu Training: {training_time:.2f}s").pack(anchor="w")
        ttk.Label(stats_frame, text=f"Tingkat Keberhasilan: {success_rate:.2%}").pack(anchor="w")
        
        # Action counts
        ttk.Label(stats_frame, text="Jumlah Aksi:").pack(anchor="w")
        for action, count in action_counts.items():
            ttk.Label(stats_frame, text=f"  {action}: {count}").pack(anchor="w")
    
    def _save_environment(self):
        """Save the current environment state."""
        if not self.env:
            messagebox.showerror("Error", "Tidak ada environment yang bisa disimpan")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filename:
            self.env.save_state(filename)
            messagebox.showinfo("Berhasil", "Environment berhasil disimpan!")
    
    def _load_environment(self):
        """Load a saved environment state."""
        filename = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.env = GridWorld.load_state(filename)
                self._update_grid_from_env()
                messagebox.showinfo("Berhasil", "Environment berhasil diload!")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal load environment: {str(e)}")
    
    def _save_q_table(self):
        """Save the Q-table to a file."""
        # First check if environment exists
        if not self.env:
            messagebox.showerror("Error", "Buat environment dulu sebelum simpan Q-table!")
            return
            
        # Then check if we have a trained agent
        if not hasattr(self, 'current_agent') or self.current_agent is None:
            messagebox.showerror("Error", "Train agent dulu sebelum simpan Q-table!")
            return
        
        # Get file path from user
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Get Q-table data
                q_table = self.current_agent.get_q_table()
                policy = self.current_agent.get_policy(self.env.size)
                
                # Prepare data to save
                data = {
                    'q_table': q_table.tolist(),  # Convert numpy array to list
                    'policy': policy.tolist(),
                    'env_size': self.env.size,
                    'method': self.training_method.get(),
                    'alpha': ALPHA,
                    'gamma': GAMMA,
                    'epsilon': EPSILON
                }
                
                # Save to JSON file
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=4)
                
                messagebox.showinfo("Berhasil", "Q-table berhasil disimpan!")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal simpan Q-table: {str(e)}")
    
    def _load_q_table(self):
        """Load a Q-table from a file."""
        # First check if environment exists
        if not self.env:
            messagebox.showerror("Error", "Buat environment dulu sebelum load Q-table!")
            return
        
        # Get file path from user
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Load data from JSON file
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                # Check if environment size matches
                if data['env_size'] != self.env.size:
                    messagebox.showerror("Error", 
                        f"Ukuran environment tidak cocok! (Loaded: {data['env_size']}, Current: {self.env.size})")
                    return
                
                # Create new agent with loaded Q-table
                self.current_agent = Agent(
                    state_space_size=self.env.get_state_space_size(),
                    action_space_size=self.env.get_action_space_size(),
                    alpha=data['alpha'],
                    gamma=data['gamma'],
                    epsilon=data['epsilon'],
                    method=data['method']
                )
                
                # Set Q-table
                self.current_agent.set_q_table(np.array(data['q_table']))
                
                # Update UI to show loaded policy
                self._update_policy_display(np.array(data['policy']))
                
                messagebox.showinfo("Berhasil", "Q-table berhasil diload!")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal load Q-table: {str(e)}")
    
    def _update_policy_display(self, policy: np.ndarray):
        """Update the grid display to show the loaded policy."""
        if not hasattr(self, 'grid_buttons') or not self.env:
            return
        
        # Clear existing policy display
        for i in range(self.env.size):
            for j in range(self.env.size):
                if self.env.map[i, j] not in [GridWorld.HOLE, GridWorld.GOAL]:
                    self.grid_buttons[i][j].config(text="E")
        
        # Show policy arrows
        arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        for i in range(self.env.size):
            for j in range(self.env.size):
                if self.env.map[i, j] not in [GridWorld.HOLE, GridWorld.GOAL]:
                    action = policy[i, j]
                    self.grid_buttons[i][j].config(text=arrows[action])
    
    def _update_grid_from_env(self):
        """Update the grid display from the current environment."""
        if not self.env:
            return
        
        self.size.set(self.env.size)
        self._create_grid()
        
        # Update positions
        self.start_pos = self.env.start_pos
        self.goal_pos = self.env.goal_pos
        self.holes = self.env.holes
        
        # Update grid buttons
        for i in range(self.env.size):
            for j in range(self.env.size):
                pos = (i, j)
                if pos == self.start_pos:
                    self.grid_buttons[i][j].config(text=GridWorld.START)
                elif pos in self.goal_pos:
                    self.grid_buttons[i][j].config(text=GridWorld.GOAL)
                elif pos in self.holes:
                    self.grid_buttons[i][j].config(text=GridWorld.HOLE)
                else:
                    self.grid_buttons[i][j].config(text=GridWorld.FROZEN)
    
    def _create_grid(self):
        """Create or update the grid display."""
        # Clear existing grid
        for row in self.grid_buttons:
            for button in row:
                button.destroy()
        self.grid_buttons.clear()
        
        # Reset environment variables
        self.start_pos = None
        self.goal_pos = []
        self.holes.clear()
        
        # Create new grid
        size = self.size.get()
        for i in range(size):
            row_buttons = []
            for j in range(size):
                btn = tk.Button(self.grid_frame, text="E", width=4, height=2,
                              command=lambda r=i, c=j: self._on_grid_click(r, c))
                btn.grid(row=i, column=j, padx=1, pady=1)
                row_buttons.append(btn)
            self.grid_buttons.append(row_buttons)
        
        # Create initial environment
        self.env = GridWorld(
            size=size,
            start_pos=None,
            goal_pos=[],
            holes=[],
            is_slippery=self.is_slippery.get()
        )
    
    def _on_grid_click(self, row: int, col: int):
        """Handle grid cell clicks."""
        tile = self.selected_tile.get()
        pos = (row, col)
        
        # Handle special tiles
        if tile == GridWorld.START:
            if self.start_pos:
                old_r, old_c = self.start_pos
                self.grid_buttons[old_r][old_c].config(text="E")
            self.start_pos = pos
        elif tile == GridWorld.GOAL:
            if pos in self.goal_pos:
                self.goal_pos.remove(pos)
            else:
                self.goal_pos.append(pos)
        elif tile == GridWorld.HOLE:
            if pos not in self.holes:
                self.holes.append(pos)
        elif tile == GridWorld.FROZEN:
            if pos == self.start_pos:
                self.start_pos = None
            elif pos in self.goal_pos:
                self.goal_pos.remove(pos)
            elif pos in self.holes:
                self.holes.remove(pos)
        
        # Update button text
        if pos in self.goal_pos:
            self.grid_buttons[row][col].config(text=GridWorld.GOAL)
        else:
            self.grid_buttons[row][col].config(text=tile)
            
        # Update environment
        self.env = GridWorld(
            size=self.size.get(),
            start_pos=self.start_pos,
            goal_pos=self.goal_pos,
            holes=self.holes,
            is_slippery=self.is_slippery.get()
        )
    
    def _create_environment(self) -> Optional[GridWorld]:
        """
        Create a new GridWorld environment based on current UI state.
        
        Returns:
            Optional[GridWorld]: The created environment or None if creation fails
        """
        try:
            # Get current UI state
            size = self.size.get()
            is_slippery = self.is_slippery.get()
            
            # Create environment with current state
            env = GridWorld(
                size=size,
                start_pos=self.start_pos or (0, 0),  # Default to (0,0) if not set
                goal_pos=self.goal_pos or [(size-1, size-1)],  # Default to bottom-right if not set
                holes=self.holes or [],  # Default to empty list if not set
                is_slippery=is_slippery
            )
            
            return env
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create environment: {str(e)}")
            return None
    
    def _update_plot_loop(self):
        """Periodically check for new plots to display."""
        try:
            # Only update if we have new data
            if not self.plot_queue.empty():
                data = self.plot_queue.get_nowait()
                
                # Only create new window if needed
                if not self.plot_window or self.plot_window.is_closed:
                    self.plot_window = ComparisonPlotWindow(self.root)
                
                # Check if we have data for both methods
                if 'sarsa' in data and 'qlearning' in data:
                    self.plot_window.update_plot(data['sarsa'], data['qlearning'])
                
                self.plot_queue.task_done()
                
        except queue.Empty:
            pass
        finally:
            # Only schedule next update if training is active
            if not self.stop_training:
                self.root.after(500, self._update_plot_loop)  # Reduced update frequency
    
    def _train_agent(self, method: str, stats: dict, shared_data: dict):
        """Train an agent using the specified method."""
        if not self.env:
            messagebox.showerror("Error", "Please create an environment first!")
            return
        
        # Create agent with current hyperparameters
        agent = Agent(
            state_space_size=self.env.get_state_space_size(),
            action_space_size=self.env.get_action_space_size(),
            alpha=self.alpha.get(),
            gamma=self.gamma.get(),
            epsilon=self.epsilon.get(),
            method=method
        )
        
        # Training loop
        rewards = []
        steps = []
        start_time = time.time()
        last_plot_update = 0
        plot_update_interval = 1.0  # Update plot every second
        
        for episode in range(self.episodes.get()):
            if self.stop_training:
                break
                
            state = self.env.reset()
            total_reward = 0
            step_count = 0
            
            while True:
                action = agent.choose_action(state, self.env.size)
                next_state, reward, done, _ = self.env.step(action)
                
                if method == "sarsa":
                    next_action = agent.choose_action(next_state, self.env.size)
                    agent.update(state, action, reward, next_state, next_action, done, self.env.size)
                else:  # q_learning
                    agent.update(state, action, reward, next_state, done=done, grid_size=self.env.size)
                
                state = next_state
                total_reward += reward
                step_count += 1
                
                if done:
                    break
            
            rewards.append(total_reward)
            steps.append(step_count)
            
            # Update shared data for plotting
            current_time = time.time()
            if current_time - last_plot_update >= plot_update_interval:
                shared_data[method] = (rewards.copy(), self.env, agent.get_policy(self.env.size))
                last_plot_update = current_time
                
                # Update stats every 10 episodes
                if episode % 10 == 0:
                    stats['time'] = current_time - start_time
                    stats['counts'] = agent.get_action_counts()
                    stats['rate'] = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                    
                    # Update stats in UI thread
                    stats_frame = (self.sarsa_stats_frame if method == 'sarsa' 
                                 else self.qlearning_stats_frame)
                    self.root.after(0, lambda: self._update_stats(
                        stats_frame, stats['time'], stats['counts'], stats['rate']
                    ))
                    
                    # Update plot if we have data for both methods
                    if 'sarsa' in shared_data and 'qlearning' in shared_data:
                        self.plot_queue.put(shared_data.copy())
                
                time.sleep(0.1)  # Small delay to prevent overwhelming the queue
    
    def _toggle_training(self):
        """Start or stop training for both methods."""
        if (self.sarsa_thread and self.sarsa_thread.is_alive()) or \
           (self.qlearning_thread and self.qlearning_thread.is_alive()):
            self.stop_training = True
            self.train_button.config(text="Stop Training")
        else:
            self.stop_training = False
            self.train_button.config(text="Stop Training")
            
            # Create shared data dictionary for thread communication
            shared_data = {}
            
            # Start both training threads
            self.sarsa_thread = threading.Thread(
                target=self._train_agent,
                args=('sarsa', self.sarsa_stats, shared_data)
            )
            self.qlearning_thread = threading.Thread(
                target=self._train_agent,
                args=('qlearning', self.qlearning_stats, shared_data)
            )
            
            self.sarsa_thread.daemon = True
            self.qlearning_thread.daemon = True
            
            self.sarsa_thread.start()
            self.qlearning_thread.start()
    
    def run(self):
        """Start the UI."""
        self.root.mainloop()

if __name__ == "__main__":
    app = FrozenLakeUI()
    app.run() 