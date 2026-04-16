import main  # noqa: F401  # Ensure the model registry is loaded.

from app.api import routes
from app.utils.validation import OSISInput, SensitivityInput, SimulationInput


BASE_PAYLOAD = {
    "laser_wavelength_nm": 405,
    "numerical_aperture": 0.85,
    "spot_size_nm": 290.47,
    "track_pitch_nm": 320,
    "layer_count": 2,
    "layer_spacing_nm": 25000,
    "isi_factor": 0.92,
    "crosstalk_factor": 0.11,
    "recording_material": "GST_HTL",
    "thermal_conductivity_w_mk": 1.5,
    "activation_energy_ev": 2.0,
    "temperature_c": 25,
    "relative_humidity": 45,
    "prml_enabled": 1,
    "ctc_enabled": 1,
}


def test_predict_snr_returns_frontend_fields():
    data = routes.predict_snr(OSISInput(**BASE_PAYLOAD))

    assert "predicted_snr_db" in data
    assert "snr_lower_bound_db" in data
    assert "snr_upper_bound_db" in data
    assert "shap_explanations" in data
    assert isinstance(data["shap_explanations"], list)


def test_sensitivity_and_dashboard_return_renderable_frames():
    sensitivity = routes.sensitivity_analysis(
        SensitivityInput(**BASE_PAYLOAD, modulation="OOK-NRZ", delta_fraction=0.05)
    )
    assert isinstance(sensitivity["ranked_sensitivity"], list)
    assert sensitivity["ranked_sensitivity"]
    assert "parameter" in sensitivity["ranked_sensitivity"][0]

    dashboard = routes.simulate_dashboard(
        SimulationInput(
            base_config=OSISInput(**BASE_PAYLOAD),
            sweep_parameter="numerical_aperture",
            start=0.78,
            end=0.9,
            steps=6,
            modulation="OOK-NRZ",
        )
    )
    assert dashboard["status"] == "success"
    assert len(dashboard["frames"]) == 6
    assert {"value", "predicted_snr_db", "estimated_ber"} <= set(dashboard["frames"][0])
