# ============================================================
# main.py — Streamlit entrypoint (cloud-safe)
# ============================================================

import streamlit as st
from pathlib import Path


# ------------------------------------------------------------
# Secrets / Config
# ------------------------------------------------------------
DEMO_MODE = st.secrets.get("DEMO_MODE", True)
MAX_TIME = st.secrets.get("MAX_TIME_SECONDS", 30)

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*missing ScriptRunContext.*"
)


# ============================================================
# main.py — Streamlit Cloud entrypoint
# ============================================================

# import streamlit as st
import subprocess
import sys
# from pathlib import Path

st.set_page_config(
    page_title="Optimization Simulator",
    layout="wide",
)

st.title("Optimization Simulator")
st.markdown(
    "Execution wrapper for a consolidated OR-Tools solver."
)

BASE_DIR = Path(__file__).parent
SOLVER_PATH = BASE_DIR / "solver" / "MPythonORToolsV9_6.py"

st.info("The solver is executed as an isolated process.")

if st.button("Run Optimization"):

    with st.spinner("Running solver..."):

        result = subprocess.run(
            [sys.executable, str(SOLVER_PATH)],
            capture_output=True,
            text=True,
        )

    st.success("Solver finished")

    # --- STDOUT ---
    if result.stdout:
        st.subheader("Solver output")
        st.code(result.stdout)

    # --- STDERR ---
    if result.stderr:
        st.subheader("Solver warnings / errors")
        st.code(result.stderr)
