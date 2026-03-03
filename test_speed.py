import urllib.request
import json
import time

url = "http://localhost:8000/optimize_parameters"
data = {
  "base_config": {
    "laser_wavelength_nm": 405,
    "numerical_aperture": 0.85,
    "spot_size_nm": 290.4,
    "track_pitch_nm": 225.0,
    "layer_count": 1,
    "layer_spacing_nm": 20000.0,
    "isi_factor": 1.29,
    "crosstalk_factor": 0.0,
    "recording_material": "GST_HTL",
    "thermal_conductivity_w_mk": 1.5,
    "activation_energy_ev": 2.0,
    "temperature_c": 25.0,
    "relative_humidity": 45.0,
    "prml_enabled": 1,
    "ctc_enabled": 1
  },
  "modulation": "OOK-NRZ",
  "top_k": 5
}

start = time.time()
req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as response:
        end = time.time()
        print(f"Success! Time taken: {end - start:.2f} seconds")
        res_data = json.loads(response.read().decode())
        print(f"Evaluated candidates: {res_data.get('evaluated_candidates')}")
        print(f"Top recommendation SNR: {res_data.get('top_recommendations')[0].get('predicted_snr_db'):.2f} dB")
except urllib.error.HTTPError as e:
    print(f"Error: {e.code} - {e.read().decode()}")
except Exception as e:
    print(f"Error: {e}")
