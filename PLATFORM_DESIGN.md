# Comprehensive Optical Storage Device Design & Simulation Platform

This document outlines the extended architecture, parameter definitions, and simulation modules required to upgrade the Interactive SNR Predictor into a comprehensive, industry-aligned optical storage simulation framework.

## 1. EXTENDED PARAMETER GROUPS

### A. Disc Physical Stack (Thin-Film Engineering)
Represents the precise multilayer structure of the optical media, essential for modeling reflectivity and thermal behavior.

| Parameter | Unit | Default | Purpose / Description |
| :--- | :--- | :--- | :--- |
| **Substrate Material** | Enum | Polycarbonate | Base material holding the stack (Polycarbonate, Glass, PMMA). |
| **Substrate Thickness** | mm | 1.2, 0.6, 0.1 | Defines baseline mechanical stability and optical path length. |
| **Dielectric 1 / 2 Material** | Enum | ZnS-SiO2 | Optical matching & thermal buffering layers. |
| **Dielectric 1 / 2 Thickness**| nm | 100 / 20 | Controls heat flow and optical interference via TMM calculation. |
| **Reflective Layer Material**| Enum | Ag-Alloy | Heat sink and main reflective boundary (Al, Ag, Au). |
| **Reflective Layer Thickness**| nm | 120 | Controls maximum reflectivity and radial cooling rate. |
| **Phase-Change Layer Thick.**| nm | 15 | Active recording layer (e.g., Ge2Sb2Te5 - GST). |
| **Protective Coating Thick.**| nm | 10000 | Prevents oxidation and physical damage. |
| **Adhesive Layer Properties**| - | UV-cured | Bonding layers for dual/multi-layer discs. |
| **Layer Roughness**| nm | 0.5 | Used for surface scattering loss calculations. |
| **Refractive Index $(n, k)$** | - | Material dependent | Complex refractive index for each layer at operating $\lambda$. |

**Data Structure for Multilayer Stack:**
```json
{
  "stack": [
    {"layer": "substrate", "material": "polycarbonate", "d_nm": 1.2e6, "n": 1.58, "k": 0.0},
    {"layer": "dielectric1", "material": "ZnS-SiO2", "d_nm": 100, "n": 2.1, "k": 0.0},
    {"layer": "active", "material": "GST", "d_nm": 15, "n": 4.1, "k": 2.1},
    {"layer": "dielectric2", "material": "ZnS-SiO2", "d_nm": 20, "n": 2.1, "k": 0.0},
    {"layer": "reflective", "material": "Ag", "d_nm": 120, "n": 0.05, "k": 4.0}
  ]
}
```
*Validation Rules*: Thickness components must be $> 0$; $\sum$ thickness must not exceed optical / mechanical operating envelopes for the selected system standard.

---

### B. Laser & Optical Pickup System
Controls the physics of the optical head used for read/write operations.

| Parameter | Unit | Default | Purpose / Description |
| :--- | :--- | :--- | :--- |
| **Read/Write Laser Power** | mW | Read: 1, Write: 10 | Operating optical power for reading and marking phase changes. |
| **Pulse Duration** | ns | 50 | Base unit of time for writing a single mark. |
| **Pulse Shape** | Enum | Gaussian | Time-domain profile of the write pulse. |
| **Beam Profile** | Enum | TEM00 | Spatial intensity distribution of the incident laser. |
| **Obj. Lens Focal Length** | mm | 3.0 | Determines the working numerical distance. |
| **Spot Ellipticity** | Ratio | 1.0 | Ratio of x to y axis of the laser spot; 1.0 = purely circular. |
| **Focus/Tracking Offset** | μm | 0.0 | Simulates static or dynamic misalignment errors. |
| **Photodetector Sensitivity** | A/W | 0.4 | Conversion of optical power to electrical current. |
| **Signal Amplification Gain** | dB | 20 | Internal transimpedance amplifier (TIA) gain applied to readout. |
| **Write Strategy** | Enum | Multi-pulse | Specifies adaptive power grouping (e.g. multi-pulse, cooling gaps, castle). |

---

### C. Thermal & Phase-Change Modeling
Physics parameters calculating the thermodynamic conversion of the active media.

| Parameter | Unit | Default | Purpose / Description |
| :--- | :--- | :--- | :--- |
| **Thermal Diffusion Coeff.** | $m^2/s$ | $1.5 \times 10^{-7}$ | Rate of heat spread radially and axially through the material. |
| **Heat Capacity ($C_p$)** | $J/(kg\cdot K)$ | 200 | Energy required to raise temperature. |
| **Melting Temp ($T_m$)** | K | 890 | GST phase transition threshold (crystalline $\rightarrow$ amorphous). |
| **Crystallization Temp ($T_c$)**| K | 420 | Amorphous $\rightarrow$ crystalline phase threshold. |
| **Min. Cooling Rate** | $K/ns$ | $>10$ | Minimum required thermal quench gradient to trap amorphous phase. |

**Key Equations:**
1. **Heat Diffusion (3D representation):**
   $$ \rho C_p \frac{\partial T}{\partial t} = \nabla \cdot (\kappa \nabla T) + Q(r, z, t) $$
   *(where $Q$ is laser absorption source term, $\kappa$ is thermal conductivity, $\rho$ is material density)*
   
2. **Phase Probability (Johnson-Mehl-Avrami-Kolmogorov formulation):**
   $$ \chi(t) = 1 - \exp(-K(T)t^n) $$
   *(where $K(T) = K_0 \exp(-E_a / k_BT)$ describing the Arrhenius rate behavior)*

---

### D. Data Encoding & Error Correction
Signal processing boundaries connecting analog physics with digital reliability.

| Parameter | Unit | Default | Purpose / Description |
| :--- | :--- | :--- | :--- |
| **Channel Coding** | Enum | EFMPlus | Conversion map of logical bytes out to channel constraint bits. |
| **Modulation Scheme** | Enum | NRZI | Defines how bits map to physical transitions on the material. |
| **ECC Type** | Enum | Reed-Solomon | Error Correction strategies (RS for DVD, LDPC/RS for advanced). |
| **Interleaving Depth** | bytes | 16 | Depth of interleaving designed to defeat burst media errors. |
| **Sector Size** | bytes | 2048 | Logical payload isolated inside one physical sector. |
| **Sync Patterns** | bits | 14-bit | Synchronization headers providing necessary PLL timing recovery. |
| **Jitter Tolerance** | % (clock) | 8.0% | Max allowable timing shift before bit boundaries are violated. |

**Signal Quality Pipeline:**
$$ \text{SNR}_{\text{channel}} \xrightarrow[\text{Viterbi}/\text{PRML}]{\text{Equalization}} \text{Raw BER} \xrightarrow{\text{ECC Logic}} \text{Post-FEC BER (User Reliability)} $$

---

### E. Servo & Mechanical System
Macro-scale dynamics introducing mechanical variances to the read/write process.

| Parameter | Unit | Default | Purpose / Description |
| :--- | :--- | :--- | :--- |
| **Spindle Speed** | RPM | 200 - 10000 | Rotational speed (important for Constant Angular Velocity). |
| **Velocity Mode** | Enum | CLV | Constant Linear Velocity vs Constant Angular Velocity. |
| **Track Following Error** | nm | $\pm 10$ | Direct contributor to Adjacent Track Crosstalk (XT). |
| **Focus Error Signal (FES)** | μm | $\pm 0.1$ | Beam defocus translating to increased Spot Size / ISI. |
| **Actuator Response Time** | ms | 1.0 | Defines latency for optical head corrections. |
| **Vibration Noise** | $m/s^2$ | 0.5 | Environmental physical shock amplitude. |
| **Disc Tilt Angle** | deg | 0.1 | Induces asymmetrical Coma aberrations in the laser spot. |

---

### F. Manufacturing Process Parameters
Determines baseline limits of the medium before simulation starts.

| Parameter | Unit | Default | Purpose / Description |
| :--- | :--- | :--- | :--- |
| **Molding Temperature** | °C | 350 | Injecting polycarbonate; affects disc birefringence. |
| **Mold Pressure** | tons | 50 | Affects pit/groove replication fidelity. |
| **Cooling Time** | s | 2.5 | Affects molecular stress and flatness (warp). |
| **Sputtering Rate** | nm/s | 5.0 | Influences grain size of the thin-film layers. |
| **Cleanroom Class** | ISO | Class 100 | Direct predictor of surface defect distributions. |
| **Defect Density** | $\text{def}/cm^2$| 0.05 | Baseline error rate for missing/deformed data marks. |
| **Expected Yield** | % | 98 | Production yield estimation logic parameter. |

---

### G. Standards & Compliance Modes
Auto-adjusting presets that pre-fill fields according to recognized specifications:

| Mode | $\lambda$ (nm) | NA | Track Pitch (nm) | Substrate Thick (mm) | User Capacity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CD** | 780 | 0.45 | 1600 | 1.2 | 700 MB |
| **DVD** | 650 | 0.60 | 740 | 0.6 | 4.7 GB |
| **Blu-ray** | 405 | 0.85 | 320 | 0.1 | 25 GB |

---

## 2. BACKEND SIMULATION MODEL PIPELINE

The core simulation is modular, constructed via isolated process blocks ensuring data flows linearly from physics to digital reliability.

1. **Optical Model:** 
    * *Inputs:* $\lambda$, NA, Laser Power, Focal Length, Defocus, Tilt.
    * *Processing:* Vector diffraction theory calculation.
    * *Outputs:* 3D Spot Intensity Profile $I(x,y,z)$.
2. **Thin-Film Optics:**
    * *Inputs:* Structure layer stack ($n, k, d$).
    * *Processing:* Transfer Matrix Method (TMM) to resolve boundary conditions.
    * *Outputs:* Overall Reflection ($R$), Transmission ($T$), Absorption probability density $A(z)$.
3. **Thermal Model (Write Simulation):**
    * *Inputs:* $I(x,y,z)$, $A(z)$, Write Strategy, $C_p$, $\kappa$, Velocity.
    * *Processing:* Finite Difference Time Domain (FDTD) or FEM heat solver.
    * *Outputs:* Dynamic temperature field $T(x,y,z,t)$, resultant state Phase map (Amorphous vs Crystalline).
4. **Signal Model (Readback):**
    * *Inputs:* Active material phase map, Read Laser Profile.
    * *Processing:* Optical signal convolution along the track.
    * *Outputs:* Direct High Frequency (HF) readout voltage waveform $V(t)$.
5. **Noise Model:**
    * *Inputs:* $V(t)$, Shot noise baseline, thermal noise from electronics, medium roughness scatter.
    * *Processing:* Adds modeled $N(t)$ forming a noisy output trace $V_{noisy}(t)$.
6. **Channel Model:**
    * *Inputs:* $V_{noisy}(t)$, Track Pitch, FES/TES.
    * *Processing:* Generates logical interference patterns (ISI) & adds neighbor track data (Crosstalk).
7. **Decoder Model:**
    * *Inputs:* Channel signal, Reference clock, PRML constraints.
    * *Processing:* Equalization, Phase-Locked Loop (PLL) synchronization, Viterbi traceback detection.
    * *Outputs:* Digitally recovered bitstream.
8. **BER Estimator:**
    * *Inputs:* Recovered bitstream vs Original encoded generator bitstream.
    * *Processing:* Hardware-simulated XOR logic operation.
    * *Outputs:* Statistically validated actual BER.

---

## 3. ADVANCED VISUALIZATIONS

High-quality web dashboard representations:

1. **SNR vs NA & $\lambda$:** Interactive 3D Surface / Heatmap defining operational boundaries.
2. **BER vs Laser Power (Threshold Curve):** Bathtub plots displaying under-power margins (failure to melt/crystallize) vs over-power margins (cross-erase and thermal bloom).
3. **Temperature vs Time:** 2D comparative line graph observing varying nodes in the disc (center of active mark vs edge of track) through a multi-pulse write strategy.
4. **Layer Reflectivity vs Wavelength:** Optical spectrum line plot from 300nm up to 900nm for the specific multi-layer dielectric stack defined by the user.
5. **Simulated Eye Diagram:** Crucial readout characteristic display. Stacked high-frequency waveforms showing amplitude eye-opening, structural jitter boundaries, and noise floors.
6. **Jitter Histogram:** Distribution mapping the precise timing distances of 0-to-1 transitions overlaid with continuous Gaussian approximation curves.

---

## 4. UI/UX STRUCTURE

A clean, engineering-focused interface divided symmetrically:

**Layout Design:**
* **Left Sidebar (Navigation):** Categorized system blocks (1. Multi-Layer Stack, 2. Optical Unit, 3. Servo/Process, 4. Code & Channel).
* **Main Left Panel (Input Accordions):** Dropdowns and collapsible grids for parameters.
  * *Features:* Integrated input sliders binding accurately with numeric text boxes. Validation lock toggles (preventing physically impossible structures). Comprehensive hover tooltips providing immediate physics summaries per metric.
* **Top Header:** Global scenario configurations (CD, DVD, Blu-Ray presets), primary `Run Complete Simulation` execution button, and `Export/Load JSON Config` features.
* **Main Right Panel (Output & Analytics):** Tabulated visualization frames.
  * *Tab 1: Dashboard Stats* (Current BER, Derived SNR, Max Junction Temp).
  * *Tab 2: Optical Layer Profiles* (Stack reflectivity graph).
  * *Tab 3: Write Cycle Analysis* (Thermal pulse charting).
  * *Tab 4: Readback Verification* (Eye diagrams and Jitter histograms).

---

## 5. IMPLEMENTATION DETAILS

### Recommended Tech Stack
* **Frontend:** React / Next.js. State handled via Zustand (allows deep object mutations required by complex parameters) handling all parameter syncs. Charts built with ECharts or Plotly.js to manage high data-density 3D matrices or heavy Eye Diagrams.
* **Backend:** FastAPI for asynchronous endpoints. NumPy & SciPy for matrix optical computations. PyTorch (optional) if attempting to utilize GPU-accelerated tensor convolution for the computationally dense thermal FDTD solver. 

### JSON Configuration Schema
Strict definition mapping ensuring validation bounds before calculations start.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CompleteOSISConfiguration",
  "type": "object",
  "properties": {
    "stackConfig": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "layerName": {"type": "string"},
          "thicknessNm": {"type": "number", "minimum": 0},
          "material": {"type": "string"},
          "refractiveIndexN": {"type": "number"},
          "extinctionCoefficientK": {"type": "number"}
        },
        "required": ["layerName", "thicknessNm", "material", "refractiveIndexN"]
      }
    },
    "opticalConfig": {
      "type": "object",
      "properties": {
        "wavelengthNm": {"type": "number", "minimum": 300, "maximum": 1200},
        "numericalAperture": {"type": "number", "minimum": 0.1, "maximum": 0.99},
        "laserPowerWriteMw": {"type": "number"}
      }
    },
    "servoConfig": {
      "type": "object",
      "properties": {
        "spindleVelocityRpm": {"type": "number"},
        "focusOffsetUm": {"type": "number"}
      }
    }
  },
  "required": ["stackConfig", "opticalConfig"]
}
```

### Simulation Execution Flow (Example API Request)

Execution utilizes a single payload aggregating all module data into one synchronous or web-socket streaming endpoint.

**Request:** `POST /api/v2/simulate_platform`
```json
{
  "opticalConfig": {
    "wavelengthNm": 405, 
    "numericalAperture": 0.85, 
    "laserPowerWriteMw": 8.0
  },
  "stackConfig": [
    {"layerName": "ZnS-SiO2", "thicknessNm": 20, "material": "Dielectric"},
    {"layerName": "GST", "thicknessNm": 15, "material": "PhaseChange"}
  ],
  "thermalConfig": {"ambientTempK": 298, "thermalDiffCoeff": 1.5e-7},
  "options": {"generateEyeDiagram": true, "generateThermalTimeSeries": false}
}
```

**Response Payload:**
```json
{
  "status": "success",
  "pipelineMetrics": {
    "snrDb": 24.5,
    "berPostFec": 1.2e-6,
    "crosstalkRatioDb": -32.1,
    "maxSpotTempK": 910
  },
  "visualizations": {
    "eyeDiagramData": {
      "timeBaseVector": [0.0, 0.1, 0.2, ...],
      "voltageSweeps": [[...], [...], [...]]
    },
    "reflectivitySpectrum": {
      "wavelengths": [300, 310, 320, ...],
      "reflectivityPercentage": [40.1, 40.5, 41.2, ...]
    }
  }
}
```
