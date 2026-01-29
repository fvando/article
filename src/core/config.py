import os
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "ml" / "models"

# App Settings
APP_TITLE = "Driver Schedule Solver"
DEMO_MODE = True
MAX_TIME_SECONDS = 30 # Default solver limit

# Visualization
FIG_HEIGHT = 4.5

# Solver / Business Logic Constants
SLOT_MINUTES = 15
SLOTS_PER_HOUR = 60 // SLOT_MINUTES

# Shift Constraints (in slots)
SHIFT_MIN_SLOTS = 36   # 9h
SHIFT_MAX_SLOTS = 52   # 13h

# Penalties and Rewards (Big M)
PENALTY_UNMET_DEMAND = 1_000_000
PENALTY_ACTIVE_DRIVER = 10_000
REWARD_TASK_SERVED = 1.0

# Reguations (Standard EU)
DAILY_LIMIT_HOURS = 9.0
WEEKLY_LIMIT_HOURS = 56.0
BIWEEKLY_LIMIT_HOURS = 90.0

# Credentials (Simple Auth)
ADMIN_USER = "admin"
ADMIN_PASS = "123456"
