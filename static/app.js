function calculatePhysics() {
    const wl = parseFloat(document.getElementById("wavelength").value);
    const na = parseFloat(document.getElementById("na").value);
    const pitch = parseFloat(document.getElementById("track_pitch").value);

    if (wl && na) {
        const spotSize = (0.61 * wl) / na;
        document.getElementById("spot_size").value = spotSize.toFixed(2);

        if (pitch) {
            const isi = spotSize / pitch;
            document.getElementById("isi").value = isi.toFixed(4);
            const crosstalk = Math.exp(-0.002 * (pitch - spotSize));
            document.getElementById("crosstalk").value = crosstalk.toExponential(4);
        }
    }
}

function applySweepDefaults() {
    const param = document.getElementById("sweep_param").value;
    const startEl = document.getElementById("sim_start");
    const endEl = document.getElementById("sim_end");

    const defaults = {
        numerical_aperture: { start: 0.75, end: 0.92 },
        track_pitch_nm: { start: 180, end: 500 },
        temperature_c: { start: 20, end: 80 },
        relative_humidity: { start: 10, end: 90 },
        laser_wavelength_nm: { start: 405, end: 780 }
    };

    startEl.value = defaults[param].start;
    endEl.value = defaults[param].end;
}

function buildPayload() {
    return {
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
}

async function postJson(url, payload) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        const detail = await response.text();
        throw new Error(`${url} failed: ${detail}`);
    }
    return response.json();
}

let snrChart = null;
let simulationChart = null;

function renderSensitivitySweep(simFrames) {
    const labels = simFrames.map(f => Number(f.value).toFixed(3));
    const snrValues = simFrames.map(f => Number(f.predicted_snr_db).toFixed(3));
    const ctx = document.getElementById("snrChart").getContext("2d");

    if (snrChart) snrChart.destroy();

    snrChart = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [{
                label: "Predicted SNR (dB)",
                data: snrValues,
                borderColor: "#34f5ff",
                backgroundColor: "rgba(52, 245, 255, 0.2)",
                tension: 0.35,
                borderWidth: 2,
                pointRadius: 3
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { labels: { color: "#dce7ff" } } },
            scales: {
                x: { ticks: { color: "#b8c5ef" }, grid: { color: "rgba(255,255,255,0.08)" } },
                y: { ticks: { color: "#b8c5ef" }, grid: { color: "rgba(255,255,255,0.08)" } }
            }
        }
    });
}

function renderSimulationChart(simFrames, param) {
    const labels = simFrames.map(f => Number(f.value).toFixed(3));
    const snr = simFrames.map(f => f.predicted_snr_db);
    const ber = simFrames.map(f => f.estimated_ber);
    const ctx = document.getElementById("simulationChart").getContext("2d");

    if (simulationChart) simulationChart.destroy();

    simulationChart = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "Hybrid SNR (dB)",
                    data: snr,
                    borderColor: "#9a6bff",
                    backgroundColor: "rgba(154, 107, 255, 0.2)",
                    yAxisID: "y",
                    tension: 0.3,
                    borderWidth: 2,
                    pointRadius: 2
                },
                {
                    label: "Estimated BER",
                    data: ber,
                    borderColor: "#34d399",
                    backgroundColor: "rgba(52, 211, 153, 0.18)",
                    yAxisID: "y1",
                    tension: 0.3,
                    borderWidth: 2,
                    pointRadius: 2
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: "#dce7ff" } },
                title: {
                    display: true,
                    text: `Real-Time Simulation Sweep (${param})`,
                    color: "#dce7ff"
                }
            },
            scales: {
                x: { ticks: { color: "#b8c5ef" }, grid: { color: "rgba(255,255,255,0.08)" } },
                y: {
                    type: "linear",
                    position: "left",
                    ticks: { color: "#b8c5ef" },
                    grid: { color: "rgba(255,255,255,0.08)" }
                },
                y1: {
                    type: "logarithmic",
                    position: "right",
                    ticks: { color: "#9ddfbe" },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });
}

function renderOptimizationTable(rows) {
    const body = document.getElementById("optimizationBody");
    body.innerHTML = "";
    rows.forEach((row, idx) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${idx + 1}</td>
            <td>${Number(row.numerical_aperture).toFixed(3)}</td>
            <td>${Number(row.track_pitch_nm).toFixed(2)}</td>
            <td>${Number(row.temperature_c).toFixed(1)}</td>
            <td>${Number(row.relative_humidity).toFixed(1)}</td>
            <td>${Number(row.predicted_snr_db).toFixed(3)}</td>
            <td>${Number(row.estimated_ber).toExponential(3)}</td>
        `;
        body.appendChild(tr);
    });
}

function renderSensitivityList(items) {
    const container = document.getElementById("sensitivityBody");
    container.innerHTML = "";
    items.slice(0, 7).forEach(item => {
        const div = document.createElement("div");
        div.className = "sensitivity-item";
        div.innerHTML = `
            <span>${item.parameter}</span>
            <span>${Number(item.normalized_sensitivity).toFixed(4)}</span>
        `;
        container.appendChild(div);
    });
}

function updateMetricCards(snrData, berData, comparisonData) {
    document.getElementById("physicsSnrValue").textContent = `${Number(snrData.physics_snr_db).toFixed(3)} dB`;
    
    // Add UQ Confidence Intervals
    const hybridHtml = `
      ${Number(snrData.predicted_snr_db).toFixed(3)} dB
      <div style="font-size: 0.8rem; color: #9a6bff; margin-top: 4px;">
        90% CI: [${Number(snrData.snr_lower_bound_db).toFixed(2)}, ${Number(snrData.snr_upper_bound_db).toFixed(2)}]
      </div>
    `;
    document.getElementById("hybridSnrValue").innerHTML = hybridHtml;
    
    document.getElementById("berValue").textContent = Number(berData.estimated_ber).toExponential(3);
    document.getElementById("gainValue").textContent = `${Number(comparisonData.snr_gain_over_analytical_db).toFixed(3)} dB`;
}

function renderShap(shapData) {
    const container = document.getElementById("shapBody");
    if(!container) return;
    container.innerHTML = "";
    shapData.forEach(item => {
        const div = document.createElement("div");
        div.className = "sensitivity-item";
        
        // Color code positive vs negative impact
        const impactColor = item.impact > 0 ? '#34d399' : '#f43f5e';
        const sign = item.impact > 0 ? '+' : '';
        
        div.innerHTML = `
            <span>${item.feature}</span>
            <span style="color: ${impactColor}; font-weight: 600;">${sign}${Number(item.impact).toFixed(4)} dB</span>
        `;
        container.appendChild(div);
    });
}

function renderComparisonText(cmpData) {
    const hasMeasured = cmpData.measured_snr_db !== undefined;
    const measuredLine = hasMeasured
        ? `Measured SNR: ${Number(cmpData.measured_snr_db).toFixed(3)} dB | Analytical Error: ${Number(cmpData.abs_error_analytical_db).toFixed(3)} dB | Hybrid Error: ${Number(cmpData.abs_error_ml_hybrid_db).toFixed(3)} dB`
        : "Measured SNR not provided.";

    document.getElementById("comparisonText").textContent =
        `Analytical SNR: ${Number(cmpData.analytical_physics_snr_db).toFixed(3)} dB | Hybrid SNR: ${Number(cmpData.ml_hybrid_snr_db).toFixed(3)} dB | BER Reduction Ratio: ${Number(cmpData.ber_reduction_ratio).toFixed(3)} | ${measuredLine}`;
}

const triggerInputs = ["wavelength", "na", "track_pitch", "layer_spacing"];
triggerInputs.forEach(id => document.getElementById(id).addEventListener("input", calculatePhysics));
document.getElementById("sweep_param").addEventListener("change", applySweepDefaults);

calculatePhysics();
applySweepDefaults();

document.getElementById("osisForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const btn = document.querySelector(".submit-btn");
    const originalText = btn.textContent;
    btn.textContent = "Running analysis...";
    btn.disabled = true;

    try {
        const basePayload = buildPayload();
        const modulation = document.getElementById("modulation").value;
        const measuredInput = document.getElementById("measured_snr").value.trim();
        const topK = parseInt(document.getElementById("top_k").value) || 3;
        const deltaFraction = parseFloat(document.getElementById("delta_fraction").value) || 0.05;
        const sweepParameter = document.getElementById("sweep_param").value;
        const simStart = parseFloat(document.getElementById("sim_start").value);
        const simEnd = parseFloat(document.getElementById("sim_end").value);
        const simSteps = parseInt(document.getElementById("sim_steps").value) || 20;

        const comparisonPayload = { ...basePayload, modulation };
        if (measuredInput !== "") {
            comparisonPayload.measured_snr_db = parseFloat(measuredInput);
        }

        const [snrData, berData, cmpData, optData, sensData, simData] = await Promise.all([
            postJson("/predict_snr", basePayload),
            postJson("/predict_ber", { ...basePayload, modulation }),
            postJson("/compare_models", comparisonPayload),
            postJson("/optimize_parameters", { base_config: basePayload, modulation, top_k: topK }),
            postJson("/sensitivity_analysis", { ...basePayload, modulation, delta_fraction: deltaFraction }),
            postJson("/simulate_dashboard", {
                base_config: basePayload,
                sweep_parameter: sweepParameter,
                start: simStart,
                end: simEnd,
                steps: simSteps,
                modulation
            })
        ]);

        updateMetricCards(snrData, berData, cmpData);
        renderOptimizationTable(optData.top_recommendations || []);
        renderSensitivityList(sensData.ranked_sensitivity || []);
        renderShap(snrData.shap_explanations || []);
        renderComparisonText(cmpData);
        renderSensitivitySweep(simData.frames || []);
        renderSimulationChart(simData.frames || [], sweepParameter);

        const simMeta = document.getElementById("simMeta");
        simMeta.textContent = `${simData.frames.length} frames generated for ${sweepParameter} sweep from ${simStart} to ${simEnd}.`;

        const resultDiv = document.getElementById("result");
        resultDiv.style.display = "block";
        if (typeof lenis !== 'undefined') {
            lenis.scrollTo(resultDiv, { offset: -100 });
        } else {
            resultDiv.scrollIntoView({ behavior: "smooth" });
        }
    } catch (error) {
        console.error(error);
        alert("Analysis failed. Ensure backend is running and inputs are valid.");
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
});
