import os

# -----------------------------
# Dataset Location
# -----------------------------
CSV_PATH = os.getenv("CSV_PATH", "Iris.csv")

# -----------------------------
# User Goal for EDA
# -----------------------------
USER_GOAL = (
    "Perform exploratory data analysis on the dataset to identify patterns, "
    "missing values, potential outliers, and provide a structured EDA proposal."
)

# -----------------------------
# Where the final proposal will be saved
# (High-risk action tool will write here after human approval)
# -----------------------------
DEFAULT_SAVE_PATH = os.getenv("DEFAULT_SAVE_PATH", "outputs/final_eda_proposal.txt")

# -----------------------------
# Persistent memory checkpoint database
# (LangGraph checkpointer will store session state here)
# -----------------------------
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "agent_state/checkpoints.sqlite")

# -----------------------------
# Default thread/session identifier
# (Used for session recovery)
# -----------------------------
DEFAULT_THREAD_ID = "eda-session-iris"