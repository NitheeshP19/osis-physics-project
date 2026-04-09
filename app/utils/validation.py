# app/utils/validation.py
from pydantic import BaseModel, root_validator, Field
from typing import Any, Dict, List, Optional
from app.utils.constants import NA_MIN, NA_MAX, TRACK_PITCH_MIN, TRACK_PITCH_MAX, TEMP_MIN, TEMP_MAX

class OSISInput(BaseModel):
    laser_wavelength_nm: int = Field(..., gt=0, le=1500, description="Laser wavelength must be > 0")
    numerical_aperture: float = Field(..., gt=0, lt=1, description="Numerical aperture between 0 and 1")
    spot_size_nm: float = Field(default=0.0, ge=0)
    track_pitch_nm: float = Field(..., gt=0)
    layer_count: int = Field(..., gt=0, le=10)
    layer_spacing_nm: float = Field(default=0.0, ge=0)
    isi_factor: float = Field(default=0.0, ge=0)
    crosstalk_factor: float = Field(default=0.0, ge=0)
    recording_material: str
    thermal_conductivity_w_mk: float = Field(..., gt=0)
    activation_energy_ev: float = Field(..., gt=0)
    temperature_c: float = Field(..., ge=-100, le=200)
    relative_humidity: float = Field(..., ge=0, le=100)
    prml_enabled: int = Field(..., ge=0, le=1)
    ctc_enabled: int = Field(..., ge=0, le=1)

class BERInput(OSISInput):
    modulation: str = "OOK-NRZ"

class ComparisonInput(OSISInput):
    modulation: str = "OOK-NRZ"
    measured_snr_db: Optional[float] = None

class OptimizationInput(BaseModel):
    base_config: OSISInput
    modulation: str = "OOK-NRZ"
    top_k: int = Field(5, ge=1, le=50)

class SensitivityInput(OSISInput):
    delta_fraction: float = Field(0.05, gt=0, le=0.5)
    modulation: str = "OOK-NRZ"

class SimulationInput(BaseModel):
    base_config: OSISInput
    sweep_parameter: str = "numerical_aperture"
    start: float
    end: float
    steps: int = Field(20, ge=5, le=100)
    modulation: str = "OOK-NRZ"

class BatchSimulationItem(BaseModel):
    clientId: str = Field(..., min_length=1, max_length=100)
    label: Optional[str] = None
    config: OSISInput
    modulation: Optional[str] = None

class BatchSimulationRequest(BaseModel):
    batchConfigs: List[BatchSimulationItem] = Field(..., min_length=1, max_length=25)
    modulation: str = "OOK-NRZ"

# Advanced Platform Payload definition
class StackLayerConfig(BaseModel):
    layerName: str
    thicknessNm: float = Field(..., gt=0)
    material: str
    refractiveIndexN: float = Field(default=1.5, gt=0)
    extinctionCoefficientK: float = Field(default=0.0, ge=0)

class OpticalConfig(BaseModel):
    wavelengthNm: float = Field(..., gt=0)
    numericalAperture: float = Field(..., gt=0, lt=1)
    laserPowerWriteMw: float = Field(..., gt=0, le=50)

class ThermalConfig(BaseModel):
    ambientTempK: float = Field(default=298.15, gt=0)
    thermalDiffCoeff: float = Field(default=1.5e-7, gt=0)

class ManufacturingConfig(BaseModel):
    moldingTempC: float = Field(default=350.0, gt=100)
    moldPressureTons: float = Field(default=50.0, gt=10)
    coolingTimeS: float = Field(default=2.5, gt=0)
    sputteringRateNmS: float = Field(default=5.0, gt=0)
    defectDensity: float = Field(default=0.05, ge=0)
    baseThicknessNm: float = Field(default=15.0, gt=0)
    refractiveIndexN1: float = Field(default=1.5, gt=0)
    refractiveIndexN2: float = Field(default=4.1, gt=0)
    refractiveIndexK2: float = Field(default=2.1, ge=0)
    thicknessVariationScale: float = Field(default=0.05, ge=0)

class AdvancedSimInput(BaseModel):
    opticalConfig: OpticalConfig
    stackConfig: List[StackLayerConfig]
    thermalConfig: Optional[ThermalConfig] = ThermalConfig()
    manufacturingConfig: Optional[ManufacturingConfig] = ManufacturingConfig()
    servoConfig: Optional[Dict[str, Any]] = {}
    options: Optional[Dict[str, bool]] = {}
    simulationMode: str = Field(default="fast", pattern="^(fast|manufacturing)$")
