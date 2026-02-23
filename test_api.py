import urllib.request
import json
import time
import subprocess

# Start server
p = subprocess.Popen([r".\.venv\Scripts\python.exe", "-m", "uvicorn", "main:app", "--port", "8000"])
time.sleep(3)

url = "http://localhost:8000/predict_snr"
data = {
    'laser_wavelength_nm': 405,
    'numerical_aperture': 0.85,
    'spot_size_nm': 290.47,
    'track_pitch_nm': 225,
    'layer_count': 1,
    'layer_spacing_nm': 20000,
    'isi_factor': 1.29,
    'crosstalk_factor': 0.0,
    'recording_material': 'GST_HTL',
    'thermal_conductivity_w_mk': 1.5,
    'activation_energy_ev': 2.0,
    'temperature_c': 25,
    'relative_humidity': 45,
    'prml_enabled': 1,
    'ctc_enabled': 1
}

req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as response:
        print("Response:", response.read().decode())
except Exception as e:
    print("Error:", e)
    if hasattr(e, 'read'):
        print(e.read().decode())

p.terminate()
