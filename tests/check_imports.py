import sys
import os
from pathlib import Path
import importlib.util

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

print(f"Checking imports from {BASE_DIR}...")

try:
    print("Importing config...")
    from src.core import config
    
    print("Importing auth...")
    from src.app import auth
    
    print("Importing helper modules...")
    from src.solver import heuristic
    print("  - heuristic OK")
    from src.solver import kpis
    print("  - kpis OK")
    from src.solver import dataset_builder
    print("  - dataset_builder OK")
    from src.solver import lns
    print("  - lns OK")
    from src.vis import charts
    print("  - charts OK")
    from src.ml import ml_guidance
    print("  - ml_guidance OK")
    
    print("Checking src.app.simulation availability...")
    spec = importlib.util.find_spec("src.app.simulation")
    if spec is None:
        raise ImportError("Could not find src.app.simulation")
    print("  - src.app.simulation found (not executed to avoid Streamlit context errors)")
    
    print("\n✅ ALL IMPORTS OK.")
except ImportError as e:
    print(f"\n❌ IMPORT ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
