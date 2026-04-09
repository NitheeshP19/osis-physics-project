import asyncio

import main  # noqa: F401  # Ensure the model registry is loaded.
from app.services.batch_simulation import run_batch_simulation
from app.utils.validation import BatchSimulationItem


def make_config(
    client_id: str,
    label: str,
    numerical_aperture: float,
    track_pitch_nm: float,
    temperature_c: float,
    humidity: float,
):
    return BatchSimulationItem(
        clientId=client_id,
        label=label,
        modulation="OOK-NRZ",
        config={
            "laser_wavelength_nm": 405,
            "numerical_aperture": numerical_aperture,
            "spot_size_nm": 290.47,
            "track_pitch_nm": track_pitch_nm,
            "layer_count": 2,
            "layer_spacing_nm": 25000,
            "isi_factor": 0.92,
            "crosstalk_factor": 0.11,
            "recording_material": "GST_HTL",
            "thermal_conductivity_w_mk": 1.5,
            "activation_energy_ev": 2.0,
            "temperature_c": temperature_c,
            "relative_humidity": humidity,
            "prml_enabled": 1,
            "ctc_enabled": 1,
        },
    )


def test_batch_simulation_service_ranks_results():
    batch_configs = [
        make_config("sim-a", "Simulation 1", 0.80, 330, 30, 55),
        make_config("sim-b", "Simulation 2", 0.88, 300, 24, 42),
    ]

    result = asyncio.run(run_batch_simulation(batch_configs, default_modulation="OOK-NRZ"))

    assert result["total_simulations"] == 2
    assert len(result["rankedResults"]) == 2
    assert result["optimal_config"]["rank"] == 1
    assert [item["rank"] for item in result["rankedResults"]] == [1, 2]
    assert result["rankedResults"][0]["optimization_score"] >= result["rankedResults"][1]["optimization_score"]
    assert "recording_material" in result["rankedResults"][0]["inputConfig"]
    assert "hybrid_snr_db" in result["rankedResults"][0]
    assert "post_fec_ber" in result["rankedResults"][0]
