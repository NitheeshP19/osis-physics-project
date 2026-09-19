# Core Physical Concepts

OSIS models the complete optical storage readout channel through four coupled physical stages:

```
Physical Disc Stack & Laser Source
               │
               ▼
   [Stage 1: Multilayer Optics]       ──> Abelès Transfer Matrix Method (TMM)
               │                          Yields R_land, R_mark, and Optical Contrast
               ▼
  [Stage 2: Diffraction & MTF]        ──> Rayleigh Spot & Circular Pupil Autocorrelation
               │                          Yields Spot Radius and MTF Attenuation M
               ▼
   [Stage 3: Optoelectronics]         ──> Photodiode Responsivity & Optical Coupling
               │                          Yields Photocurrent & Signal Power P_sig
               ▼
     [Stage 4: Noise Channel]         ──> Shot Noise + Laser RIN + Thermal Noise
               │                          Yields Total Noise Power P_noise
               ▼
        [Output Metrics]              ──> CNR (dB) and Bit Error Rate (BER)
```

---

## 1. Multilayer Interference (Thin-Film Optics)
Optical discs store data via structural or crystalline phase differences:
- **Land state:** Unwritten or crystalline state of the phase-change alloy (e.g., $\mathrm{Ge}_2\mathrm{Sb}_2\mathrm{Te}_5$).
- **Mark state:** Written or amorphous pit state formed by laser quench pulses.

Because layer thicknesses are on the order of tens of nanometers (comparable to the read wavelength), multiple reflections between layer boundaries generate optical interference. OSIS computes the exact reflection coefficients by multiplying the $2 \times 2$ Abelès characteristic matrices of each layer.

---

## 2. Spatial Resolution (Diffraction & MTF)
The read laser is focused through an objective lens with numerical aperture $\mathrm{NA} = n \sin\theta$.
The focal spot forms an Airy disc of radius:
$$r_{\mathrm{spot}} = 0.61 \frac{\lambda}{\mathrm{NA}}$$

When recorded marks have spatial periodicity approaching the optical resolution limit, the high-frequency Fourier components of the mark pattern fall outside the lens pupil. The detected signal modulation is attenuated by the incoherent Modulation Transfer Function (MTF):
$$M = \mathrm{MTF}(\nu_{\mathrm{mark}})$$
Signal power scales as $M^2$.

---

## 3. Optoelectronic Detection
The reflected light is captured onto a PIN photodiode with quantum efficiency $\eta_q$. The photodiode converts absorbed photons into photocurrent:
$$\mathcal{R}(\lambda) = \frac{\eta_q e}{E_{\mathrm{ph}}} = \frac{\eta_q e \lambda}{h c} \quad [\mathrm{A/W}]$$

---

## 4. Noise Physics
Three fundamental physical noise sources limit detection:
1. **Quantum Shot Noise:** Fundamental Poisson fluctuations in photon arrivals ($2 e I_{\mathrm{dc}} B$).
2. **Relative Intensity Noise (RIN):** Technical intensity fluctuations of the laser cavity ($\mathrm{RIN} \cdot I_{\mathrm{dc}}^2 B$).
3. **Johnson–Nyquist Thermal Noise:** Thermal agitation of charge carriers inside the load resistor ($4 k_B T B / R_L$).
