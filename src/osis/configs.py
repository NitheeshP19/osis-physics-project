"""
Disc configuration presets for OSIS.

Each ``DiscConfig`` bundles all parameters needed to run a simulation: optical-head
parameters, detector parameters, and thin-film layer stacks for land and mark states.

Parameter values are sourced from published standards and peer-reviewed literature;
sources are cited inline.  Values marked **approximate** are representative of
typical devices and may differ from specific hardware implementations.

Physical Disc Standards used as sources
----------------------------------------
- CD:      Ecma International, *Standard ECMA-130*, 2nd ed., 1996.
- DVD:     Ecma International, *Standard ECMA-267*, 4th ed., 2008.
- Blu-ray: Blu-ray Disc Association, *System Description Blu-ray Disc Read-Only
           Format*, Part 1 (Physical Specifications), v1.0, 2004.

Material optical constants
---------------------------
- Ag reflector: Rakic (1998). *Applied Optics*, 37(22), 5271–5283.
- ZnS–SiO2 dielectric: Ohta et al. (1998). *Jpn. J. Appl. Phys.*, 37, 2247.
- GST (Ge2Sb2Te5): Yamada et al. (1991). *Jpn. J. Appl. Phys.*, 30, 49.
  Values are wavelength-specific and distinguish amorphous (mark) from
  crystalline (land) phases. Given the sensitivity of optical constants to
  deposition conditions and composition, these values are representative.
  Users may override them for specific material systems via ``land_stack`` /
  ``mark_stack`` arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class ThinFilmLayer:
    """One layer in a multilayer thin-film stack.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. ``"ZnS-SiO2"``).
    thickness_m : float
        Physical thickness in metres. Must be > 0.
    n : float
        Real part of the complex refractive index (≥ 1 for physical materials).
    k : float
        Imaginary part of the complex refractive index (extinction coefficient ≥ 0).
        Positive k implies absorption.
    source : str, optional
        Bibliographic reference for the optical constants.
    """
    name: str
    thickness_m: float
    n: float
    k: float
    source: str = ""

    def __post_init__(self) -> None:
        if self.thickness_m <= 0:
            raise ValueError(f"Layer thickness must be > 0, got {self.thickness_m}")
        if self.n < 0:
            raise ValueError(f"Refractive index n must be >= 0, got {self.n}")
        if self.k < 0:
            raise ValueError(f"Extinction coefficient k must be >= 0, got {self.k}")


@dataclass
class DiscConfig:
    """Complete configuration for an optical-storage readout simulation.

    Users should normally instantiate one of the standard presets (``CDConfig``,
    ``DVDConfig``, ``BluRayConfig``) rather than this class directly.

    Attributes
    ----------
    name : str
        Human-readable disc identifier.
    wavelength_m : float
        Laser wavelength in metres.
    numerical_aperture : float
        Objective lens NA (dimensionless, 0 < NA < 1).
    track_pitch_m : float
        Track pitch (centre-to-centre spacing) in metres.
    min_mark_m : float
        Shortest recorded mark (pit) length in metres.
    laser_power_w : float
        Read laser power at the disc surface in watts.
    coupling_efficiency : float
        Fraction of reflected signal captured by the detector (0–1).
        Accounts for optical losses in the return path.
    detector_bandwidth_hz : float
        Detection bandwidth in Hz (determines noise floor).
    rin_per_hz : float
        Relative Intensity Noise spectral density in 1/Hz.
        Typical semiconductor laser: ~1e-15 to 1e-14 /Hz.
    load_resistance_ohm : float
        Transimpedance amplifier equivalent load resistance in ohms.
    temperature_k : float
        Operating temperature in kelvin (affects Johnson noise).
    quantum_efficiency : float
        Photodetector quantum efficiency (0–1).
    land_stack : list[ThinFilmLayer]
        Ordered thin-film layer stack for the land (unrecorded/crystalline) state,
        from incident medium outward (deepest layer last). The incident medium
        (typically air, n=1) and substrate are NOT included in this list;
        they are specified separately in the TMM solver via ``n_incident`` and
        ``n_substrate``.
    mark_stack : list[ThinFilmLayer]
        Ordered thin-film layer stack for the mark (pit/amorphous) state.
    n_incident : float
        Real refractive index of the incident medium (default 1.0 for air).
    n_substrate : complex
        Complex refractive index of the substrate (default polycarbonate n=1.58).
    """
    name: str
    wavelength_m: float
    numerical_aperture: float
    track_pitch_m: float
    min_mark_m: float
    laser_power_w: float
    coupling_efficiency: float
    detector_bandwidth_hz: float
    rin_per_hz: float
    load_resistance_ohm: float
    temperature_k: float
    quantum_efficiency: float
    land_stack: list[ThinFilmLayer] = field(default_factory=list)
    mark_stack: list[ThinFilmLayer] = field(default_factory=list)
    n_incident: float = 1.0
    n_substrate: complex = 1.58 + 0j   # polycarbonate

    def __post_init__(self) -> None:
        if not (0 < self.numerical_aperture < 1):
            raise ValueError(
                f"numerical_aperture must satisfy 0 < NA < 1, got {self.numerical_aperture}"
            )
        if self.wavelength_m <= 0:
            raise ValueError(f"wavelength_m must be > 0, got {self.wavelength_m}")
        if self.laser_power_w <= 0:
            raise ValueError(f"laser_power_w must be > 0")


# ---------------------------------------------------------------------------
# CD — Compact Disc Read-Only Memory
# Source: Ecma-130, 2nd ed. (1996); Bouwhuis et al. (1985), "Principles of
# Optical Disc Systems", Adam Hilger.
# ---------------------------------------------------------------------------
def CDConfig(**kwargs) -> DiscConfig:  # noqa: N802
    """Return a configuration preset for a Compact Disc (CD).

    Optical constants for the Al reflector at 780 nm are from
    Palik (1985), *Handbook of Optical Constants of Solids*.

    The simplified single-layer model represents a typical pressed CD
    (read-only) with an aluminium reflector. Recorded (mark) state uses a
    slightly reduced reflectivity to represent a dye or pressed pit.

    All parameter values are representative of the standard and may not match
    specific hardware implementations.
    """
    defaults = dict(
        name="CD",
        wavelength_m=780e-9,          # Ecma-130: 780 nm
        numerical_aperture=0.45,       # Ecma-130: 0.45
        track_pitch_m=1.6e-6,          # Ecma-130: 1.6 µm
        min_mark_m=0.833e-6,           # Ecma-130: T3 = 0.833 µm (1×)
        laser_power_w=0.5e-3,          # typical read power (approximate)
        coupling_efficiency=0.35,      # typical OPU return-path efficiency
        detector_bandwidth_hz=50e6,    # sufficient for 1× data rate ~1.23 Mbit/s
        rin_per_hz=1e-14,              # typical semiconductor laser
        load_resistance_ohm=500.0,
        temperature_k=300.0,
        quantum_efficiency=0.80,
        n_incident=1.0,
        n_substrate=1.58 + 0j,        # polycarbonate
        # CD-RW phase-change stack (incident -> substrate):
        # ZnS-SiO2 (100 nm) | AgInSbTe active (20 nm) | ZnS-SiO2 (20 nm) | Al reflector (70 nm)
        # Land = crystalline state; Mark = amorphous state
        land_stack=[
            ThinFilmLayer("ZnS-SiO2", 100e-9, 2.12, 0.0, "Ohta et al. (1998)"),
            ThinFilmLayer("AgInSbTe-crystalline", 20e-9, 4.40, 2.10, "Tomiyoshi et al. (1998), Jpn. J. Appl. Phys."),
            ThinFilmLayer("ZnS-SiO2", 20e-9, 2.12, 0.0, "Ohta et al. (1998)"),
            ThinFilmLayer("Al", 70e-9, 2.80, 8.45, "Palik (1985), Handbook of Optical Constants"),
        ],
        mark_stack=[
            ThinFilmLayer("ZnS-SiO2", 100e-9, 2.12, 0.0, "Ohta et al. (1998)"),
            ThinFilmLayer("AgInSbTe-amorphous", 20e-9, 4.20, 1.20, "Tomiyoshi et al. (1998)"),
            ThinFilmLayer("ZnS-SiO2", 20e-9, 2.12, 0.0, "Ohta et al. (1998)"),
            ThinFilmLayer("Al", 70e-9, 2.80, 8.45, "Palik (1985)"),
        ],
    )
    defaults.update(kwargs)
    return DiscConfig(**defaults)


# ---------------------------------------------------------------------------
# DVD — Digital Versatile Disc
# Source: Ecma-267, 4th ed. (2008)
# ---------------------------------------------------------------------------
def DVDConfig(**kwargs) -> DiscConfig:  # noqa: N802
    """Return a configuration preset for a DVD (DVD-RW / DVD-RAM).

    Uses a Ge2Sb2Te5 (GST) phase-change recording stack with Ag reflector;
    optical constants at 650 nm from Yamada et al. (1991) and Rakic (1998).
    """
    defaults = dict(
        name="DVD",
        wavelength_m=650e-9,           # Ecma-267: 650 nm
        numerical_aperture=0.60,        # Ecma-267: 0.60
        track_pitch_m=0.74e-6,          # Ecma-267: 0.74 µm
        min_mark_m=0.4e-6,              # Ecma-267: 0.4 µm (1×)
        laser_power_w=0.5e-3,
        coupling_efficiency=0.35,
        detector_bandwidth_hz=100e6,
        rin_per_hz=1e-14,
        load_resistance_ohm=500.0,
        temperature_k=300.0,
        quantum_efficiency=0.80,
        n_incident=1.0,
        n_substrate=1.58 + 0j,
        # DVD-RW phase-change stack (incident -> substrate):
        # ZnS-SiO2 (90 nm) | GST active (15 nm) | ZnS-SiO2 (20 nm) | Ag reflector (100 nm)
        # Land = crystalline state; Mark = amorphous state
        land_stack=[
            ThinFilmLayer("ZnS-SiO2", 90e-9, 2.13, 0.0, "Ohta et al. (1998)"),
            ThinFilmLayer("GST-crystalline", 15e-9, 4.10, 3.20, "Yamada et al. (1991), Jpn. J. Appl. Phys."),
            ThinFilmLayer("ZnS-SiO2", 20e-9, 2.13, 0.0, "Ohta et al. (1998)"),
            ThinFilmLayer("Ag", 100e-9, 0.14, 3.98, "Rakic (1998), Appl. Opt. 37(22), 5271"),
        ],
        mark_stack=[
            ThinFilmLayer("ZnS-SiO2", 90e-9, 2.13, 0.0, "Ohta et al. (1998)"),
            ThinFilmLayer("GST-amorphous", 15e-9, 4.60, 1.80, "Yamada et al. (1991)"),
            ThinFilmLayer("ZnS-SiO2", 20e-9, 2.13, 0.0, "Ohta et al. (1998)"),
            ThinFilmLayer("Ag", 100e-9, 0.14, 3.98, "Rakic (1998)"),
        ],
    )
    defaults.update(kwargs)
    return DiscConfig(**defaults)


# ---------------------------------------------------------------------------
# Blu-ray Disc (BD-RE)
# Source: Blu-ray Disc Association, System Description v1.0 (2004);
#         Ohta et al. (2000), Jpn. J. Appl. Phys.
# ---------------------------------------------------------------------------
def BluRayConfig(**kwargs) -> DiscConfig:  # noqa: N802
    """Return a configuration preset for a Blu-ray Disc (BD-RE).

    The Blu-ray phase-change layer (Ge2Sb2Te5, GST) uses separate optical
    constants for the crystalline (land) and amorphous (mark) phases at 405 nm,
    representative of values reported in Yamada et al. (1991) and
    Ohta et al. (2000). These are approximate; see ``docs/assumptions.md``.

    Stack order (incident → substrate):
        air | ZnS-SiO2 (protective) | GST (active) | ZnS-SiO2 | Ag (reflector) | polycarbonate
    """
    # GST at 405 nm — representative values (approximate)
    # Crystalline phase (land): n≈3.6, k≈3.7 (Yamada et al. 1991, corrected)
    # Amorphous phase (mark):  n≈4.5, k≈1.5 (Ohta et al. 2000)
    # ZnS-SiO2 at 405 nm: n≈2.15 (Ohta et al. 1998)
    # Ag at 405 nm: n≈0.065, k≈1.88 (Rakic 1998)

    def _land_stack():
        return [
            ThinFilmLayer("ZnS-SiO2", 60e-9, 2.15, 0.0,
                          "Ohta et al. (1998), Jpn. J. Appl. Phys. 37, 2247"),
            ThinFilmLayer("GST-crystalline", 12e-9, 3.6, 3.7,
                          "Yamada et al. (1991), Jpn. J. Appl. Phys. 30, 49 (approx.)"),
            ThinFilmLayer("ZnS-SiO2", 20e-9, 2.15, 0.0,
                          "Ohta et al. (1998)"),
            ThinFilmLayer("Ag", 100e-9, 0.065, 1.88,
                          "Rakic (1998), Appl. Opt. 37(22), 5271"),
        ]

    def _mark_stack():
        return [
            ThinFilmLayer("ZnS-SiO2", 60e-9, 2.15, 0.0,
                          "Ohta et al. (1998)"),
            ThinFilmLayer("GST-amorphous", 12e-9, 4.5, 1.5,
                          "Ohta et al. (2000) (approx.)"),
            ThinFilmLayer("ZnS-SiO2", 20e-9, 2.15, 0.0,
                          "Ohta et al. (1998)"),
            ThinFilmLayer("Ag", 100e-9, 0.065, 1.88,
                          "Rakic (1998)"),
        ]

    defaults = dict(
        name="Blu-ray BD-RE",
        wavelength_m=405e-9,           # BDA spec: 405 nm
        numerical_aperture=0.85,        # BDA spec: 0.85
        track_pitch_m=0.32e-6,          # BDA spec: 0.32 µm
        min_mark_m=0.149e-6,            # BDA spec: shortest mark ≈ 0.149 µm
        laser_power_w=0.5e-3,
        coupling_efficiency=0.30,       # slightly lower due to higher NA
        detector_bandwidth_hz=200e6,
        rin_per_hz=1e-14,
        load_resistance_ohm=500.0,
        temperature_k=300.0,
        quantum_efficiency=0.75,
        n_incident=1.0,
        n_substrate=1.58 + 0j,
        land_stack=_land_stack(),
        mark_stack=_mark_stack(),
    )
    defaults.update(kwargs)
    return DiscConfig(**defaults)


#: Pre-built preset instances indexed by name.
PRESETS: dict[str, DiscConfig] = {
    "CD": CDConfig(),
    "DVD": DVDConfig(),
    "Blu-ray": BluRayConfig(),
}
