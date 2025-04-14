# Single Agent Reinforcement Learning

A Python implementation of reinforcement learning algorithms (SARSA and Q-Learning) for the Frozen Lake environment, featuring a graphical user interface for interactive training and comparison.

## Features

- Interactive grid environment creation
- Support for multiple goals and holes
- Slippery ice option for stochastic transitions
- Real-time training visualization
- Side-by-side comparison of SARSA and Q-Learning
- Training statistics and policy visualization
- Save/Load environment and Q-tables

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Tkinter (usually comes with Python)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/elderaka/Single-Agent-Reinforcement-Learning.git
cd Single-Agent-Reinforcement-Learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the UI:
```bash
python ui.py
```

2. Create your environment:
   - Set grid size
   - Place start position
   - Add goals and holes
   - Toggle slippery option

3. Training:
   - Click "Compare Methods" to start training
   - Watch real-time comparison of SARSA and Q-Learning
   - Monitor training statistics
   - Save environment or Q-tables for later use

## Project Structure

- `ui.py`: Main UI application
- `env.py`: Environment implementation
- `agent.py`: Reinforcement learning agents
- `utils.py`: Utility functions
- `config.py`: Configuration settings

## License

MIT License 