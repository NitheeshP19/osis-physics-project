# OSIS Quickstart

## Basic Simulation

```python
import osis

# Simulate Blu-ray BD-RE
config = osis.BluRayConfig()
result = osis.simulate(config)

print(f"CNR: {result['cnr_db']:.2f} dB")
print(f"BER: {result['ber']:.2e}")
print(f"Spot radius: {result['spot_radius_m']*1e9:.1f} nm")
print(f"MTF at T_min: {result['mtf']:.3f}")
```

## Parameter Sweep

```python
import numpy as np, osis

cfg = osis.BluRayConfig()
results = osis.parameter_sweep(cfg, "numerical_aperture", np.linspace(0.75, 0.90, 7))
for r in results:
    print(f"NA={r['value']:.3f}  CNR={r['cnr_db']:.2f} dB")
```

## Sensitivity Analysis

```python
import osis

cfg = osis.BluRayConfig()
sens = osis.one_at_a_time_sensitivity(cfg, delta_fraction=0.05)
ranked = osis.rank_parameters_by_influence(sens)
for param, swing in ranked[:4]:
    print(f"{param}: CNR swing = {swing:.3f} dB, elasticity = {sens[param]['elasticity']:+.3f}")
```
