# app/utils/constants.py

# ================================
# PHYSICS CONSTANTS
# ================================
K_BOLTZMANN = 8.617e-5

# Parameter ranges used in optimization sweeps
NA_MIN, NA_MAX = 0.40, 0.95
TRACK_PITCH_MIN, TRACK_PITCH_MAX = 180.0, 1800.0
TEMP_MIN, TEMP_MAX = 20.0, 80.0
HUMIDITY_MIN, HUMIDITY_MAX = 10.0, 90.0

# ================================
# PERFORMANCE LIMITS (Render Safe)
# ================================
MAX_POINTS = 500       # Maximum array size returned to frontend to prevent render lag
MAX_TRACES = 80        # Max lines for synthetic eye diagram overlaps
MAX_TIME_STEPS = 300   # Max loop steps for simulated thermal gradients
MAX_TOP_K = 20         # Maximum recommendations returned by optimization

