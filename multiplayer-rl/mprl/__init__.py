import os
import sys

# # Append OpenSpiel (included in this package's git repository) to the system path so it can be imported.
_OPEN_SPIEL_MODULES_PATH = os.path.abspath(os.path.join(__file__, "../../../dependencies/open_spiel"))
_OPEN_SPIEL_PYBIND_PATH = os.path.join(_OPEN_SPIEL_MODULES_PATH, "build/python")
sys.path.insert(1, _OPEN_SPIEL_PYBIND_PATH)
sys.path.insert(1, _OPEN_SPIEL_MODULES_PATH)

# Tune logging/results directory defaults to ~/ray_results and can be overridden with the TUNE_SAVE_DIR env variable.
TUNE_SAVE_DIR = os.getenv("TUNE_SAVE_DIR", None)
