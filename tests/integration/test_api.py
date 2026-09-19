"""Integration tests for FastAPI application endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_read_root(client: TestClient) -> None:
    """Verify root endpoint is reachable."""
    response = client.get("/")
    assert response.status_code == 200


def test_simulate_platform_endpoint(client: TestClient) -> None:
    """Verify /api/v1/simulate_platform endpoint with valid optical and thermal config."""
    payload = {
        "opticalConfig": {
            "wavelengthNm": 405.0,
            "numericalAperture": 0.85,
            "laserPowerWriteMw": 5.0,
        },
        "thermalConfig": {
            "ambientTempK": 298.15,
            "thermalDiffCoeff": 1.5e-7,
        },
        "stackConfig": [
            {
                "layerName": "ZnS-SiO2",
                "thicknessNm": 100.0,
                "material": "ZnS-SiO2",
                "refractiveIndexN": 2.1,
                "extinctionCoefficientK": 0.0,
            },
            {
                "layerName": "GST",
                "thicknessNm": 20.0,
                "material": "GST",
                "refractiveIndexN": 4.5,
                "extinctionCoefficientK": 1.8,
            },
            {
                "layerName": "Al",
                "thicknessNm": 50.0,
                "material": "Al",
                "refractiveIndexN": 1.8,
                "extinctionCoefficientK": 6.0,
            },
        ],
        "simulationMode": "fast",
    }
    response = client.post("/api/v1/simulate_platform", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "success"
    assert "pipelineMetrics" in data
    assert "snrDb" in data["pipelineMetrics"]
    assert "berPostFec" in data["pipelineMetrics"]
    assert "visualizations" in data


def test_simulate_dashboard_endpoint(client: TestClient) -> None:
    """Verify /simulate_dashboard parameter sweep endpoint."""
    payload = {
        "sweep_parameter": "numerical_aperture",
        "start": 0.60,
        "end": 0.85,
        "steps": 5,
        "modulation": "OOK-NRZ",
        "base_config": {
            "laser_wavelength_nm": 405,
            "numerical_aperture": 0.85,
            "spot_size_nm": 290.0,
            "track_pitch_nm": 320.0,
            "layer_count": 1,
            "layer_spacing_nm": 0.0,
            "isi_factor": 0.9,
            "crosstalk_factor": 0.1,
            "recording_material": "GST_HTL",
            "thermal_conductivity_w_mk": 0.5,
            "activation_energy_ev": 2.2,
            "temperature_c": 25.0,
            "relative_humidity": 50.0,
            "prml_enabled": 1,
            "ctc_enabled": 1,
        },
    }
    response = client.post("/simulate_dashboard", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "success"
    assert len(data["frames"]) == 5
    for frame in data["frames"]:
        assert "predicted_snr_db" in frame
        assert "estimated_ber" in frame
