import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define common directory paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')
WISCONSIN_DIR = os.path.join(EXPERIMENTS_DIR, 'wisconsin')
RESULTS_DIR = os.path.join(WISCONSIN_DIR, 'results')
PLOTS_DIR = os.path.join(WISCONSIN_DIR, 'plots')
LOGS_DIR = os.path.join(WISCONSIN_DIR, 'logs')

# Create directories if they don't exist
for directory in [DATA_DIR, EXPERIMENTS_DIR, WISCONSIN_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)
