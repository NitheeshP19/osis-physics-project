from main import predict_snr, OSISInput
import json

# Test input based on a typical high-performance scenario
test_input = OSISInput(
    laser_wavelength_nm=405,
    numerical_aperture=0.85,
    spot_size_nm=290.76, # Calculated roughly
    track_pitch_nm=225,
    layer_count=3,
    layer_spacing_nm=20000,
    isi_factor=1.29, # 290.76 / 225
    crosstalk_factor=0.0, # exp(-20000/225) ~ 0
    recording_material="GST_HTL",
    thermal_conductivity_w_mk=1.5,
    activation_energy_ev=2.0,
    temperature_c=25,
    relative_humidity=45,
    prml_enabled=1,
    ctc_enabled=1
)

print("Testing prediction with input:")
print(test_input.dict())

try:
    result = predict_snr(test_input)
    print("\nResult:")
    print(json.dumps(result, indent=2))
    
    # Validation logic
    if result['predicted_snr_db'] > 0:
        print("\n✅ Prediction successful and positive.")
    else:
        print("\n❌ Prediction failed (non-positive SNR).")
        
    print(f"Physics Baseline: {result['physics_snr_db']} dB")
    print(f"ML Residual: {result['ml_residual_db']} dB")
    
except Exception as e:
    print(f"\n❌ Error during prediction: {e}")
