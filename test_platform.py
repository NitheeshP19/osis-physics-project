import urllib.request
import json

url = "http://127.0.0.1:8000/api/v1/simulate_platform"
data = {
    "simulationMode": "fast",
    "opticalConfig": {
        "wavelengthNm": 405,
        "numericalAperture": 0.85,
        "laserPowerWriteMw": 8.0
    },
    "stackConfig": [
        {"layerName": "Dielectric 1", "thicknessNm": 100, "material": "ZnS-SiO2", "refractiveIndexN": 2.1},
        {"layerName": "Active Phase", "thicknessNm": 15, "material": "GST", "refractiveIndexN": 4.1, "extinctionCoefficientK": 2.1},
        {"layerName": "Dielectric 2", "thicknessNm": 20, "material": "ZnS-SiO2", "refractiveIndexN": 2.1},
        {"layerName": "Reflective", "thicknessNm": 120, "material": "Ag", "refractiveIndexN": 0.05, "extinctionCoefficientK": 4.0}
    ],
    "thermalConfig": {
        "ambientTempK": 298.15,
        "thermalDiffCoeff": 1.5e-7
    },
    "manufacturingConfig": {
        "moldingTempC": 350.0,
        "moldPressureTons": 50.0,
        "coolingTimeS": 2.5,
        "sputteringRateNmS": 5.0,
        "baseThicknessNm": 15.0,
        "refractiveIndexN2": 4.1,
        "thicknessVariationScale": 0.05
    }
}

req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as response:
        print("Success:", response.read().decode())
except Exception as e:
    print("Error:", e)
    if hasattr(e, 'read'):
        print(e.read().decode())
