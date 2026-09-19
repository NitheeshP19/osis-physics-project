# OSIS Validation Framework

OSIS implements a three-tier validation framework to ensure scientific integrity and numerical accuracy without self-referential or synthetic assumptions.

---

## Level 1: Analytical Limiting Cases

Analytical validation tests limiting physical regimes where closed-form mathematical solutions exist:

1. **Fresnel Single Interface Limit:**
   For a normal-incidence plane wave traveling from medium $n_1 = 1.0$ into bare substrate $n_2 = 1.50$, the Abelès characteristic matrix reduces identically to the single-interface Fresnel power reflectance:
   $$R = \left|\frac{n_1 - n_2}{n_1 + n_2}\right|^2 = \left(\frac{0.5}{2.5}\right)^2 = 0.040000$$
   OSIS numerical error: $< 10^{-12}$ (machine precision).

2. **Lossless Stack Energy Conservation:**
   For an arbitrary $N$-layer dielectric stack where all extinction coefficients $k_i = 0$, Maxwell's equations require exact energy conservation:
   $$R + T = 1.0$$
   OSIS verified across multilayer $\mathrm{SiO}_2 / \mathrm{TiO}_2$ stacks to within $< 10^{-14}$ absolute error.

3. **Quarter-Wave Antireflection Coating:**
   For an index-matched dielectric layer of optical thickness $n d = \lambda / 4$ on a substrate $n_s$, with $n_{\mathrm{AR}} = \sqrt{n_s}$, the destructive interference is complete:
   $$R(\lambda) \to 0$$
   OSIS yields $R < 10^{-8}$.

4. **Rayleigh Scaling:**
   Spot radius obeys strict linear scaling with wavelength: $r(2\lambda) = 2r(\lambda)$.

5. **MTF Cutoff:**
   $\mathrm{MTF}(0) = 1.0$ and $\mathrm{MTF}(\nu \ge \nu_c) = 0.0$.

---

## Level 2: Independent Numerical Cross-Checks

To avoid circular testing within the same code path, OSIS is cross-checked against independent algorithmic implementations:

1. **Heavens (1955) Recurrence vs. Abelès (1950) Transfer Matrix:**
   An independent recurrence solver calculating boundary reflections from the substrate upwards was implemented in `benchmarks/validation/numerical_validation.py`. Across a complex 4-layer absorbing phase-change disc stack ($\mathrm{ZnS}\text{-}\mathrm{SiO}_2 / \mathrm{GST} / \mathrm{ZnS}\text{-}\mathrm{SiO}_2 / \mathrm{Al}$), both methods yield identical power reflectance to machine precision ($\Delta R = 0.00$).

2. **Analytical MTF vs. Discrete 2D Pupil Autocorrelation:**
   The analytical circular pupil formula is compared against a discrete 2D spatial grid integration of two overlapping circular pupils (1024×1024 grid). The maximum discrepancy is $< 5 \times 10^{-5}$, fully attributable to discrete grid pixelation.

---

## Level 3: Domain Physical Standards Verification

OSIS standard disc presets are evaluated against international disc specifications and published literature:

| Format | Literature / Standard Reference | Standard Spot Radius | Simulated Spot Radius | Literature CNR Range | Simulated Readout CNR |
|:-------|:--------------------------------|:---------------------|:----------------------|:---------------------|:----------------------|
| **CD-RW** | ECMA-130 / Philips-Sony Red Book | 1040–1070 nm | **1057.3 nm** | 35–45 dB | **38.98 dB** |
| **DVD-RW** | ECMA-267 / DVD Forum | 650–675 nm | **660.8 nm** | 22–35 dB | **26.19 dB** |
| **Blu-ray BD-RE** | BDA System Description Part 1 | 280–305 nm | **290.6 nm** | 15–25 dB | **16.79 dB** |

---

## Running the Validation Suite

To execute all three levels in one command:

```bash
osis validate
# or: python benchmarks/validation/run_all_validations.py
```
