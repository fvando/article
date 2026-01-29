import streamlit as st
import sys
import os
from pathlib import Path

# Ensure root (c:\project\article) is in sys.path so we can import src.*
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.core.config import APP_TITLE
from src.app.auth import check_password

# Must be the first Streamlit command
st.set_page_config(layout="wide", page_title=APP_TITLE)

from src.vis.ui import apply_corporate_style

apply_corporate_style()

def main():
    if check_password():
        # Run the simulation app
        import runpy
        sim_path = BASE_DIR / "src" / "app" / "simulation.py"
        
        # We run the script in the same process
        try:
           runpy.run_path(str(sim_path))
        except SystemExit:
           pass
        except Exception as e:
           st.error(f"Error running simulation: {e}")
           st.exception(e)

if __name__ == "__main__":
    main()
