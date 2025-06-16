import sys
from pathlib import Path


def resolve_root(levels_up: int = 1):
    """
    Adds the project root directory to sys.path for script imports.

    Args:
        levels_up (int): How many directory levels to go up from the current script.
                         Default is 1 (e.g., from 'scripts/' to project root).
    """
    current_path = Path(__file__).resolve()
    root_path = current_path.parents[levels_up]
    print("Root Path: ", root_path)
    sys.path.append(str(root_path))
