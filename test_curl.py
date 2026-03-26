import urllib.request
import json

url = "http://127.0.0.1:8000/predict_snr"
data = {
    "laser_wavelength_nm": 405,
    "numerical_aperture": 0.85,
    "spot_size_nm": 290.47,
    "track_pitch_nm": 320,
    "layer_count": 2,
    "layer_spacing_nm": 15000,
    "isi_factor": 0.9077,
    "crosstalk_factor": 0.942,
    "recording_material": "GST_HTL",
    "thermal_conductivity_w_mk": 0.5,
    "activation_energy_ev": 1.5,
    "temperature_c": 25,
    "relative_humidity": 50,
    "prml_enabled": 1,
    "ctc_enabled": 1
}

req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as response:
        print("Success:", response.read().decode())
except Exception as e:
    print("Error:", e)
    if hasattr(e, 'read'):
        print(e.read().decode())
