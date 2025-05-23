import numpy as np
from typing import Tuple, Dict, Any, List, Set
import random
import pickle
import os

class GridWorld:
    """
    Frozen Lake environment for reinforcement learning.
    Agent must navigate from start to goal while avoiding holes.
    Environment has stochastic transitions (slippery ice).
    """
    
    # Tile types
    FROZEN = 'F'
    HOLE = 'H'
    START = 'S'
    GOAL = 'G'
    
    def __init__(self, size: int = 4, start_pos: Tuple[int, int] = (0, 0), 
                 goal_pos: List[Tuple[int, int]] = None, holes: List[Tuple[int, int]] = None,
                 is_slippery: bool = True):
        """
        Initialize Frozen Lake environment.
        
        Args:
            size: Grid size (size x size)
            start_pos: Starting position (row, col). If None, defaults to (0, 0)
            goal_pos: List of goal positions. If None or empty, defaults to bottom-right corner
            holes: List of hole positions. If None or empty, generates random holes
            is_slippery: Whether environment has stochastic transitions
        """
        # Validate grid size
        if size < 2:
            raise ValueError("Grid size must be at least 2x2")
        self.size = size
        
        # Set default start position if None
        if start_pos is None:
            start_pos = (0, 0)
        self._validate_position(start_pos, "start position")
        self.start_pos = start_pos
        
        # Set default goal position if None or empty
        if goal_pos is None or not goal_pos:
            goal_pos = [(size-1, size-1)]
        for pos in goal_pos:
            self._validate_position(pos, "goal position")
        self.goal_pos = goal_pos
        
        self.is_slippery = is_slippery
        
        # Define action space (up, down, left, right)
        self.actions = {
            0: (-1, 0),   # Up
            1: (1, 0),    # Down
            2: (0, -1),   # Left
            3: (0, 1)     # Right
        }
        
        # Initialize state
        self.current_pos = self.start_pos
        
        # Generate or set holes
        if holes is None or not holes:
            self.holes = self._generate_holes()
        else:
            for pos in holes:
                self._validate_position(pos, "hole position")
            self.holes = holes
        
        # Create map
        self.map = self._create_map()
        
        # State encoding mapping
        self._state_to_idx = {}
        self._idx_to_state = {}
        self._init_state_encoding()
        
    def _validate_position(self, pos: Tuple[int, int], pos_type: str) -> None:
        """Validate a position is within grid bounds."""
        if not (0 <= pos[0] < self.size and 0 <= pos[1] < self.size):
            raise ValueError(f"Invalid {pos_type}: {pos} is outside grid bounds")
            
    def _init_state_encoding(self) -> None:
        """Initialize state encoding mapping."""
        idx = 0
        for i in range(self.size):
            for j in range(self.size):
                self._state_to_idx[(i, j)] = idx
                self._idx_to_state[idx] = (i, j)
                idx += 1
                
    def state_to_idx(self, state: Tuple[int, int]) -> int:
        """Convert 2D state to index."""
        return self._state_to_idx[state]
    
    def idx_to_state(self, idx: int) -> Tuple[int, int]:
        """Convert index to 2D state."""
        return self._idx_to_state[idx]
    
    def _generate_holes(self) -> List[Tuple[int, int]]:
        """Generate random holes, ensuring they don't overlap with start or goals."""
        holes = []
        num_holes = min(self.size, (self.size * self.size) // 4)  # Max 25% of grid
        
        while len(holes) < num_holes:
            pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if (pos != self.start_pos and 
                pos not in self.goal_pos and 
                pos not in holes):
                holes.append(pos)
        
        return holes
    
    def _create_map(self) -> np.ndarray:
        """Create the map with different tile types."""
        map = np.full((self.size, self.size), self.FROZEN)
        
        # Set start and goals
        map[self.start_pos] = self.START
        for goal in self.goal_pos:
            map[goal] = self.GOAL
        
        # Set holes
        for hole in self.holes:
            map[hole] = self.HOLE
            
        return map
    
    def reset(self) -> Tuple[int, int]:
        """Reset the environment to the start position."""
        self.current_pos = self.start_pos
        return self.current_pos
    
    def _get_slippery_action(self, intended_action: int) -> int:
        """
        Simulate slippery ice by potentially changing the intended action.
        Matches the original FrozenLake environment behavior.
        
        Args:
            intended_action: The action the agent intended to take
            
        Returns:
            The actual action that will be taken
        """
        if not self.is_slippery:
            return intended_action
            
        # With 1/3 probability, the agent will move in a different direction
        if random.random() < 1/3:
            # Choose a random action (including the intended one)
            return random.choice(list(self.actions.keys()))
        return intended_action
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0-3)
            
        Returns:
            next_state: New position (row, col)
            reward: Reward for the action
            done: Whether episode is complete
            info: Additional information
        """
        # Get actual action (considering slippery ice)
        actual_action = self._get_slippery_action(action)
        movement = self.actions[actual_action]
        
        # Calculate new position
        new_row = max(0, min(self.size-1, self.current_pos[0] + movement[0]))
        new_col = max(0, min(self.size-1, self.current_pos[1] + movement[1]))
        new_pos = (new_row, new_col)
        
        # Update state
        self.current_pos = new_pos
        
        # Check if new position is a hole or goal
        done = False
        reward = -0.01  # Small negative reward for each step
        
        if new_pos in self.holes:
            done = True
            reward = -1.0  # Negative reward for falling in a hole
        elif new_pos in self.goal_pos:
            done = True
            reward = 1.0  # Positive reward for reaching any goal
        
        # Add distance-based reward shaping
        if not done:
            min_goal_dist = min(abs(new_pos[0] - g[0]) + abs(new_pos[1] - g[1]) 
                              for g in self.goal_pos)
            reward += 0.01 * (self.size - min_goal_dist) / self.size
        
        info = {
            "action": action,
            "actual_action": actual_action,
            "old_pos": self.current_pos,
            "new_pos": new_pos,
            "tile_type": self.map[new_pos]
        }
        
        return new_pos, reward, done, info
    
    def get_state_space_size(self) -> int:
        """Get the size of the state space (number of possible positions)."""
        return self.size * self.size
    
    def get_action_space_size(self) -> int:
        """Get the size of the action space."""
        return len(self.actions)
    
    def render(self) -> None:
        """Print a simple ASCII representation of the grid."""
        grid = np.copy(self.map)
        if self.current_pos != self.start_pos and self.current_pos not in self.goal_pos:
            grid[self.current_pos] = 'A'  # Agent position
        
        print("\n".join([" ".join(row) for row in grid]))
        print()
    
    def save_state(self, filename: str) -> None:
        """Save the environment state to a file."""
        state = {
            'size': self.size,
            'start_pos': self.start_pos,
            'goal_pos': self.goal_pos,
            'holes': self.holes,
            'is_slippery': self.is_slippery,
            'map': self.map,
            'state_to_idx': self._state_to_idx,
            'idx_to_state': self._idx_to_state
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load_state(cls, filename: str) -> 'GridWorld':
        """Load an environment state from a file."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"State file {filename} not found")
        
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        env = cls(
            size=state['size'],
            start_pos=state['start_pos'],
            goal_pos=state['goal_pos'],
            holes=state['holes'],
            is_slippery=state['is_slippery']
        )
        env.map = state['map']
        env._state_to_idx = state['state_to_idx']
        env._idx_to_state = state['idx_to_state']
        return env 