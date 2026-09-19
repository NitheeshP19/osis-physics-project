# OSIS Data Directory

This directory contains reference datasets and optical constants used by the OSIS simulation framework.

## Contents

- Reference refractive index ($n, k$) tables for optical storage materials (phase-change alloys, dielectric protective layers, reflective layers) sourced from peer-reviewed literature.
- Generated datasets from `generate_osis_dataset.py` for training and evaluating machine learning surrogate models.

## Provenance of Default Preset Optical Constants

The optical constants implemented in `src/osis/configs.py` are obtained from published literature:

1. **CD-RW ($780\text{ nm}$):**
   - Substrate: Polycarbonate ($n = 1.58, k = 0.0$) [Palik, 1985]
   - Phase-change layer: $\mathrm{AgInSbTe}$ [Ohta et al., 1998]
     - Crystalline ($n = 4.25, k = 2.10$)
     - Amorphous ($n = 4.00, k = 1.45$)
   - Dielectric layers: $\mathrm{ZnS}\text{--}\mathrm{SiO}_2$ ($n = 2.10, k = 0.0$)
   - Metal reflector: $\mathrm{Al}$ ($n = 2.05, k = 7.10$) [Rakić et al., 1998]

2. **DVD-RW ($650\text{ nm}$):**
   - Substrate: Polycarbonate ($n = 1.58, k = 0.0$)
   - Phase-change layer: $\mathrm{Ge}_2\mathrm{Sb}_2\mathrm{Te}_5$ (GST) [Yamada et al., 1991]
     - Crystalline ($n = 4.30, k = 1.80$)
     - Amorphous ($n = 3.90, k = 0.90$)
   - Dielectric layers: $\mathrm{ZnS}\text{--}\mathrm{SiO}_2$ ($n = 2.12, k = 0.0$)
   - Metal reflector: $\mathrm{Al}$ alloy ($n = 1.62, k = 6.00$)

3. **Blu-ray BD-RE ($405\text{ nm}$):**
   - Cover layer: Polymer resin ($n = 1.52, k = 0.0$)
   - Phase-change layer: $\mathrm{Ge}_2\mathrm{Sb}_2\mathrm{Te}_5$ [Yamada et al., 1991; Ohta et al., 1998]
     - Crystalline ($n = 4.50, k = 1.80$)
     - Amorphous ($n = 3.80, k = 1.10$)
   - Dielectric layers: $\mathrm{ZnS}\text{--}\mathrm{SiO}_2$ ($n = 2.20, k = 0.0$)
   - Metal reflector: $\mathrm{Ag}$ alloy ($n = 0.17, k = 2.05$) [Palik, 1985]
