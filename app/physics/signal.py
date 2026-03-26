# app/physics/signal.py
import numpy as np
from typing import Dict, Any
from app.utils.constants import MAX_TRACES, MAX_POINTS

def synthesize_eye_diagram(predicted_snr_db: float) -> Dict[str, list]:
    """
    Vectorized Eye Diagram Synthesis.
    Generates overlapping waveforms corresponding to 0->1 and 1->0 mark transitions.
    """
    snr_linear = 10 ** (predicted_snr_db / 20)
    noise_amplitude = min(0.4, 1.0 / snr_linear) if snr_linear > 0 else 0.5
    
    traces = min(MAX_TRACES, 40)
    points_per_trace = min(MAX_POINTS, 50)
    
    time_base = np.linspace(0, 2, points_per_trace) # 2 UI (Unit Intervals)
    
    voltage_sweeps = []
    
    for _ in range(traces):
        # Base ideal wave
        if np.random.rand() > 0.5:
            # Rise
            wave = 0.5 * (1 - np.cos(np.pi * time_base))
        else:
            # Fall
            wave = 0.5 * (1 + np.cos(np.pi * time_base))
            
        # Add Jitter (phase shift)
        phase_jitter = np.random.normal(0, noise_amplitude * 0.1)
        shift_idx = int(phase_jitter * points_per_trace)
        wave = np.roll(wave, shift_idx)
        
        # Add amplitude noise (AWGN)
        noise = np.random.normal(0, noise_amplitude * np.random.uniform(0.5, 1.0), points_per_trace)
        wave += noise
        
        # Normalize to typical readout voltage ranges (-1 to 1 V)
        wave = (wave * 2) - 1.0
        
        voltage_sweeps.append(wave.round(3).tolist())
        
    return {
        "timeBaseUI": time_base.round(3).tolist(),
        "voltageTraces": voltage_sweeps
    }
