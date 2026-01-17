"""
quick start script for training market making agent
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.train import train_agent
from datetime import datetime, timedelta

if __name__ == "__main__":
    #quick training run with improved settings
    # Using last 24 hours for better data quality (matches new default)
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)  # Changed from days=7 to hours=24

    train_agent(
        total_timesteps = 50000, 
        n_envs = 4,  # Increased from 2 for faster training
        n_steps = 512,
        learning_rate = 3e-4,
        ent_coef = 0.05,  # Increased exploration (new default)
        model_save_path = "models/market_making_ppo",
        start_time = start_time,
        end_time = end_time
    )

    