# OSIS Physical Model

## 1. Multilayer Thin-Film Reflectance (Abelès TMM)

The power reflectance of the disc recording stack is computed using the Abelès 2×2 characteristic matrix method (Born & Wolf, 2019, §1.6).

For each layer `i` with complex refractive index `ñᵢ = nᵢ + ikᵢ` and thickness `dᵢ`:

```
δᵢ = (2π/λ) · ñᵢ · dᵢ        (complex phase)

Mᵢ = [[cos(δᵢ),      -i·sin(δᵢ)/ñᵢ],
       [-i·ñᵢ·sin(δᵢ), cos(δᵢ)     ]]
```

The total characteristic matrix is `M = M₁ · M₂ · ... · Mₙ`, and power reflectance is `R = |r|²`.

Separate reflectances are computed for the **land** (crystalline, unwritten) and **mark** (amorphous, written) phase-change states.

**References:** Born & Wolf (2019); Heavens (1955).

---

## 2. Scalar Diffraction & MTF

**Spot radius (Rayleigh criterion):**

```
r_spot = 0.61 · λ / NA
```

**Incoherent MTF** at spatial frequency ν:

```
MTF(ν) = (2/π) · [arccos(ν/νc) - (ν/νc)·√(1-(ν/νc)²)]   for ν ≤ νc = 2·NA/λ
```

The MTF is evaluated at the minimum mark spatial frequency `νs = 1/(2·L_min)` to yield the channel modulation factor M.

**Note:** Scalar theory is accurate for NA ≲ 0.6. At Blu-ray NA = 0.85, vector diffraction effects cause measurable deviations not modelled here.

**References:** Goodman (2017); Hecht (2017).

---

## 3. Optoelectronic Signal

The detector responsivity is `ℛ = ηq·e / E_ph` [A/W], where:
- ηq = quantum efficiency
- e = 1.602×10⁻¹⁹ C (electron charge)
- E_ph = hc/λ (photon energy)

Signal photocurrent:

```
I_sig = ℛ · ηc · P_laser · |R_land - R_mark| · M   [A]
```

Signal electrical power: `P_sig = I_sig² · R_L`.

---

## 4. Physical Noise Channel

Three independent additive Gaussian noise sources:

| Component | Formula | Reference |
|:----------|:--------|:---------|
| Shot noise | σ²_shot = 2·e·I_dc·B | Saleh & Teich (2019), §18.5 |
| Laser RIN | σ²_RIN = RIN·I_dc²·B | Petermann (1988), §5 |
| Thermal (Johnson–Nyquist) | σ²_th = 4·kB·T·B/R_L | Johnson (1928) |

Total noise: `P_noise = (σ²_shot + σ²_RIN + σ²_th) · R_L`

---

## 5. CNR and BER

```
CNR = 10·log₁₀(P_sig / P_noise)            [dB]

BER = 0.5 · erfc(√CNR_linear / (2·√2))     [OOK-NRZ, AWGN channel]
```

**References:** Proakis & Salehi (2007), §4.2.
