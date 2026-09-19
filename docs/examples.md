# Examples and Tutorial Notebooks

OSIS provides standalone runnable Python examples and an interactive Jupyter notebook.

---

## Interactive Tutorial Notebook

- **Path:** [`examples/tutorial.ipynb`](file:///c:/Users/dell/Documents/physics/examples/tutorial.ipynb)
- **Features Demonstrated:**
  1. Single-call simulation of Blu-ray BD-RE.
  2. Cross-format performance comparison table (CD, DVD, Blu-ray).
  3. Parameter sweep over Numerical Aperture (NA) plotted with Matplotlib.
  4. One-At-a-Time sensitivity analysis and parameter elasticity rankings.

---

## Standalone Example Scripts

### Example 01: Standard Optical Storage Format Comparison
- **Path:** [`examples/01_disc_comparison.py`](file:///c:/Users/dell/Documents/physics/examples/01_disc_comparison.py)
- **Run:** `python examples/01_disc_comparison.py`
- **Output:** Prints detailed optical, signal, and noise metrics for CD-RW, DVD-RW, and Blu-ray BD-RE.

### Example 02: Multilayer Phase-Change Optical Stack Analysis
- **Path:** [`examples/02_thin_film_stack.py`](file:///c:/Users/dell/Documents/physics/examples/02_thin_film_stack.py)
- **Run:** `python examples/02_thin_film_stack.py`
- **Output:** Evaluates complex $R$, $T$, and $A$ spectra and analyzes dielectric layer thickness sensitivity.

### Example 03: ML Surrogate Speedup Benchmark
- **Path:** [`examples/03_surrogate_speedup.py`](file:///c:/Users/dell/Documents/physics/examples/03_surrogate_speedup.py)
- **Run:** `python examples/03_surrogate_speedup.py`
- **Output:** Measures prediction error (RMSE $\approx 0.17\text{ dB}$) and demonstrates sub-millisecond inference throughput ($\approx 68\ \mu\text{s}$ per sample).
