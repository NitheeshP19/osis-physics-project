// Physics Constants & Formulas
function calculatePhysics() {
    const wl = parseFloat(document.getElementById("wavelength").value);
    const na = parseFloat(document.getElementById("na").value);
    const pitch = parseFloat(document.getElementById("track_pitch").value);
    const spacing = parseFloat(document.getElementById("layer_spacing").value);

    if (wl && na) {
        // Spot Size = 0.61 * lambda / NA
        const spotSize = (0.61 * wl) / na;
        document.getElementById("spot_size").value = spotSize.toFixed(2);

        if (pitch) {
            // ISI = Spot / Pitch
            const isi = spotSize / pitch;
            document.getElementById("isi").value = isi.toFixed(4);

            // Crosstalk = exp(-spacing / pitch)
            // Note: This is a simplified view for the UI. The backend uses a more complex one if needed.
            const crosstalk = Math.exp(-spacing / pitch);
            document.getElementById("crosstalk").value = crosstalk.toExponential(4);
        }
    }
}

// Chart Instance
let snrChart = null;

async function updateChart(basePayload) {
    const ctx = document.getElementById('snrChart').getContext('2d');
    
    // Sweep NA from 0.6 to 0.95 with 0.05 step
    const naSteps = [];
    for (let i = 60; i <= 95; i += 5) {
        naSteps.push(i / 100);
    }

    const predictions = [];
    
    // Show loading state on chart? Or just wait.
    // Ideally we want to run these in parallel
    const promises = naSteps.map(na => {
        const payload = { ...basePayload, numerical_aperture: na };
        // We need to re-calculate spot size and ISI for each NA step for accurate prediction
        // The backend does this, but we need to ensure the payload is correct.
        // Actually, the backend re-calculates physics terms based on inputs (like Spot size dependent on NA),
        // BUT the input validation might require consistency.
        // Let's assume the backend re-calculates derived features like spot_size from NA/WL.
        // Looking at main.py: It calculates physics_snr and features using terms from input_dict.
        // Only spot_size_nm is passed in. We must update it!
        
        const spotSize = (0.61 * payload.laser_wavelength_nm) / na;
        payload.spot_size_nm = spotSize;
        payload.isi_factor = spotSize / payload.track_pitch_nm;
        
        return fetch("/predict_snr", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        }).then(res => res.json());
    });

    try {
        const results = await Promise.all(promises);
        const dataPoints = results.map(r => r.predicted_snr_db);

        if (snrChart) {
            snrChart.destroy();
        }

        snrChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: naSteps,
                datasets: [{
                    label: 'Predicted SNR (dB)',
                    data: dataPoints,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.2)',
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 4,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'SNR Sensitivity: Numerical Aperture Sweep',
                        color: '#94a3b8'
                    },
                    legend: {
                        labels: { color: '#f8fafc' }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Numerical Aperture (NA)', color: '#94a3b8' },
                        ticks: { color: '#94a3b8' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        title: { display: true, text: 'SNR (dB)', color: '#94a3b8' },
                        ticks: { color: '#94a3b8' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });

    } catch (err) {
        console.error("Failed to update chart:", err);
    }
}

// Attach listeners
const triggerInputs = ["wavelength", "na", "track_pitch", "layer_spacing"];
triggerInputs.forEach(id => {
    document.getElementById(id).addEventListener("input", calculatePhysics);
});

// Initial Calculation
calculatePhysics();

// Form Submission
document.getElementById("osisForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const btn = document.querySelector(".submit-btn");
    const originalText = btn.textContent;
    btn.textContent = "Calculating...";
    btn.disabled = true;

    try {
        const payload = {
            laser_wavelength_nm: parseInt(document.getElementById("wavelength").value),
            numerical_aperture: parseFloat(document.getElementById("na").value),
            spot_size_nm: parseFloat(document.getElementById("spot_size").value),
            track_pitch_nm: parseFloat(document.getElementById("track_pitch").value),
            layer_count: parseInt(document.getElementById("layer_count").value),
            layer_spacing_nm: parseFloat(document.getElementById("layer_spacing").value),
            isi_factor: parseFloat(document.getElementById("isi").value),
            crosstalk_factor: parseFloat(document.getElementById("crosstalk").value),
            recording_material: document.getElementById("material").value,
            thermal_conductivity_w_mk: parseFloat(document.getElementById("thermal_k").value),
            activation_energy_ev: parseFloat(document.getElementById("activation_e").value),
            temperature_c: parseFloat(document.getElementById("temp").value),
            relative_humidity: parseFloat(document.getElementById("humidity").value),
            prml_enabled: parseInt(document.getElementById("prml").value),
            ctc_enabled: parseInt(document.getElementById("ctc").value)
        };

        const response = await fetch("/predict_snr", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error("Prediction failed");

        const data = await response.json();

        const resultDiv = document.getElementById("result");
        const snrValue = document.getElementById("snrValue");
        
        // Update Result Display with specifics
        resultDiv.style.display = "block";
        
        // Construct detailed HTML
        snrValue.innerHTML = `
            <div style="font-size: 0.9em; color: #94a3b8; margin-bottom: 0.5rem;">
                Baseline: ${data.physics_snr_db} dB | Residual: ${data.ml_residual_db > 0 ? '+' : ''}${data.ml_residual_db} dB
            </div>
            ${data.predicted_snr_db} dB
        `;

        // Update Chart
        await updateChart(payload);

        // Scroll to result
        resultDiv.scrollIntoView({ behavior: "smooth" });

    } catch (error) {
        console.error(error);
        alert("Error connecting to the model API. Ensure the backend is running.");
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
});
