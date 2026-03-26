# app/physics/thermal.py
import numpy as np
from typing import Dict, Any
from app.utils.constants import MAX_TIME_STEPS

def approximate_thermal_pulse(thermal_config: Dict[str, Any], optical_config: Dict[str, Any]) -> Dict[str, list]:
    """
    1D Vectorized Heat diffusion heuristic.
    Simulates writing a sequence of 3 marks (multi-pulse strategy).
    """
    ambient = thermal_config.get('ambientTempK', 298.15)
    diff_coeff = thermal_config.get('thermalDiffCoeff', 1.5e-7)
    power = optical_config.get('laserPowerWriteMw', 8.0)

    steps = min(MAX_TIME_STEPS, 150)
    time_times = np.linspace(0, 100, steps) # ns
    
    # Synthesize pulse envelope (3 pulses)
    pulse = np.zeros_like(time_times)
    pulse[(time_times > 10) & (time_times < 25)] = 1
    pulse[(time_times > 35) & (time_times < 50)] = 1
    pulse[(time_times > 60) & (time_times < 75)] = 1

    # Simulate temperature gradient (integrating heat + cooling)
    # Cooling term proportional to diff_coeff
    temp = np.full_like(time_times, ambient)
    current_temp = ambient
    
    power_heat_rate = power * 8.0 # Arbitrary scaling to reach ~900K for 10mW
    cooling_rate = diff_coeff * 2e8 # Diffusion escape

    for i in range(1, steps):
        dt = time_times[i] - time_times[i-1]
        
        # Adding heat if laser is on
        heat_in = pulse[i] * power_heat_rate * dt
        
        # Heat dissipating based on Newton's law of cooling approx
        heat_out = cooling_rate * (current_temp - ambient) * dt
        
        current_temp += (heat_in - heat_out)
        current_temp = min(current_temp, 3000.0)  # Cap at 3000K (physical limit for optical media)
        temp[i] = current_temp

    # Generate edge track (cooler)
    edge_temp = ambient + (temp - ambient) * 0.4

    return {
        "timeNs": time_times.round(2).tolist(),
        "centerTempK": temp.round(1).tolist(),
        "edgeTempK": edge_temp.round(1).tolist()
    }
