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
    const baseUrl = (window.location.port === "5501") ? "http://127.0.0.1:8000" : "";
    const response = await fetch(baseUrl + url, {
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
triggerInputs.forEach(id => {
    document.getElementById(id).addEventListener("input", () => {
        const formatSelect = document.getElementById("disc_format");
        if (formatSelect) formatSelect.value = "custom";
        calculatePhysics();
    });
});

const formatEl = document.getElementById("disc_format");
if (formatEl) {
    formatEl.addEventListener("change", (e) => {
        const format = e.target.value;
        if (format === "cd") {
            document.getElementById("wavelength").value = "780";
            document.getElementById("na").value = "0.45";
            document.getElementById("track_pitch").value = "1600";
            document.getElementById("layer_spacing").value = "0";
            document.getElementById("layer_count").value = "1";
        } else if (format === "dvd") {
            document.getElementById("wavelength").value = "650";
            document.getElementById("na").value = "0.60";
            document.getElementById("track_pitch").value = "740";
            document.getElementById("layer_spacing").value = "55000";
            document.getElementById("layer_count").value = "2";
        } else if (format === "bd") {
            document.getElementById("wavelength").value = "405";
            document.getElementById("na").value = "0.85";
            document.getElementById("track_pitch").value = "320";
            document.getElementById("layer_spacing").value = "25000";
            document.getElementById("layer_count").value = "2";
        }
        calculatePhysics();
    });
}
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
        const frames = (simData && simData.frames) ? simData.frames : [];
        renderSensitivitySweep(frames);
        renderSimulationChart(frames, sweepParameter);

        const simMeta = document.getElementById("simMeta");
        if (simMeta) {
            simMeta.textContent = `${frames.length} frames generated for ${sweepParameter} sweep from ${simStart} to ${simEnd}.`;
        }

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

// ================================================================
// PLATFORM SIMULATION DASHBOARD
// Matches backend: POST /api/v1/simulate_platform
// Backend returns: pipelineMetrics + visualizations
//   pipelineMetrics keys: snrDb, berPostFec, maxSpotTempK, factoryYieldPct,
//     manufacturingMode, snrCenter, snrEdge, snrMean, snrStd,
//     yieldConfidence, reflectivityMean, reflectivityStd
//   visualizations keys: eyeDiagramData, reflectivitySpectrum,
//     thermalProfile, manufacturingProcess
//   manufacturingProcess keys: radiusMm, birefringenceNm,
//     thicknessVariancePct, variance_profile, reflectivity_profile,
//     reflectivity_mean, reflectivity_std, snr_center, snr_edge, etc.
// ================================================================
document.getElementById('runPlatformSimBtn')?.addEventListener('click', async (e) => {
    e.preventDefault();
    const btn = e.target;
    const oldText = btn.textContent;
    btn.textContent = "Processing matrices...";
    btn.disabled = true;

    try {
        // Build the payload matching AdvancedSimInput schema exactly
        const payload = {
            simulationMode: document.getElementById('simulation_mode')?.value || 'fast',
            opticalConfig: {
                wavelengthNm: parseFloat(document.getElementById('wavelength').value) || 405,
                numericalAperture: parseFloat(document.getElementById('na').value) || 0.85,
                laserPowerWriteMw: 8.0
            },
            stackConfig: [
                {layerName: "Dielectric 1", thicknessNm: 100, material: "ZnS-SiO2", refractiveIndexN: 2.1},
                {layerName: "Active Phase", thicknessNm: 15, material: "GST", refractiveIndexN: 4.1, extinctionCoefficientK: 2.1},
                {layerName: "Dielectric 2", thicknessNm: 20, material: "ZnS-SiO2", refractiveIndexN: 2.1},
                {layerName: "Reflective", thicknessNm: 120, material: "Ag", refractiveIndexN: 0.05, extinctionCoefficientK: 4.0}
            ],
            thermalConfig: {
                ambientTempK: parseFloat(document.getElementById('temp')?.value || 25) + 273.15,
                thermalDiffCoeff: 1.5e-7
            },
            manufacturingConfig: {
                moldingTempC: parseFloat(document.getElementById('molding_temp')?.value || 350),
                moldPressureTons: parseFloat(document.getElementById('mold_pressure')?.value || 50),
                coolingTimeS: parseFloat(document.getElementById('cooling_time')?.value || 2.5),
                sputteringRateNmS: parseFloat(document.getElementById('sputtering_rate')?.value || 5.0),
                baseThicknessNm: parseFloat(document.getElementById('base_thickness')?.value || 15.0),
                refractiveIndexN2: parseFloat(document.getElementById('refractive_index')?.value || 4.1),
                thicknessVariationScale: parseFloat(document.getElementById('variation_scale')?.value || 0.05)
            }
        };

        const baseUrl = (window.location.port === "5501") ? "http://127.0.0.1:8000" : "";
        const res = await fetch(baseUrl + '/api/v1/simulate_platform', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!res.ok) {
            const errText = await res.text();
            throw new Error("Simulation pipeline error: " + errText);
        }
        const data = await res.json();
        const metrics = data.pipelineMetrics;

        // ---- Update Metric Cards ----
        const el = (id) => document.getElementById(id);

        if (el('simMaxTemp')) el('simMaxTemp').textContent = metrics.maxSpotTempK + " K";
        if (el('simBer')) el('simBer').textContent = Number(metrics.berPostFec).toExponential(3);
        if (el('simYield')) el('simYield').textContent = metrics.factoryYieldPct + " %";

        if (el('simSnrMCarlo')) {
            el('simSnrMCarlo').textContent = metrics.manufacturingMode
                ? `${metrics.snrMean.toFixed(2)} \u00b1 ${metrics.snrStd.toFixed(2)} dB`
                : `${metrics.snrMean.toFixed(2)} dB (Fast)`;
        }

        if (el('simSnrRadial')) {
            el('simSnrRadial').textContent = `${metrics.snrCenter.toFixed(2)} / ${metrics.snrEdge.toFixed(2)} dB`;
        }

        if (el('simReflectMCarlo')) {
            el('simReflectMCarlo').textContent = metrics.manufacturingMode
                ? `${(metrics.reflectivityMean * 100).toFixed(2)} \u00b1 ${(metrics.reflectivityStd * 100).toFixed(2)} %`
                : `${(metrics.reflectivityMean * 100).toFixed(2)} % (Fast)`;
        }

        el('simResults').style.display = 'block';

        // ---- Plotly Layout Base ----
        const layoutBase = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: {l: 50, r: 50, t: 20, b: 40},
            font: {color: '#b8c5ef'},
            xaxis: {gridcolor: 'rgba(255,255,255,0.06)'},
            yaxis: {gridcolor: 'rgba(255,255,255,0.06)'}
        };
        const plotCfg = {responsive: true, displayModeBar: false};

        // ---- 1. Eye Diagram ----
        const eyeData = data.visualizations.eyeDiagramData;
        const eyeTraces = eyeData.voltageTraces.map(trace => ({
            x: eyeData.timeBaseUI, y: trace, mode: 'lines',
            line: {color: 'rgba(52, 245, 255, 0.2)', width: 1}, hoverinfo: 'none'
        }));
        Plotly.newPlot('plotlyEye', eyeTraces, {
            ...layoutBase, showlegend: false,
            xaxis: {...layoutBase.xaxis, title: 'Time (UI)'},
            yaxis: {...layoutBase.yaxis, title: 'Voltage (V)'}
        }, plotCfg);

        // ---- 2. Reflectivity Spectrum (TMM from optics.py) ----
        const refData = data.visualizations.reflectivitySpectrum;
        Plotly.newPlot('plotlyReflect', [{
            x: refData.wavelengths, y: refData.reflectivityPercentage,
            mode: 'lines', line: {color: '#9a6bff', width: 2},
            fill: 'tozeroy', fillcolor: 'rgba(154, 107, 255, 0.08)'
        }], {
            ...layoutBase,
            xaxis: {...layoutBase.xaxis, title: 'Wavelength (nm)'},
            yaxis: {...layoutBase.yaxis, title: 'Reflectivity (%)'}
        }, plotCfg);

        // ---- 3. Radial Reflectivity Profile (from manufacturing.py Monte Carlo) ----
        const mfgData = data.visualizations.manufacturingProcess;
        if (mfgData && mfgData.reflectivity_profile && el('plotlyReflectivityProfile')) {
            Plotly.newPlot('plotlyReflectivityProfile', [{
                x: mfgData.radiusMm,
                y: mfgData.reflectivity_profile.map(v => v * 100),
                name: 'Reflectivity (%)',
                type: 'scatter', mode: 'lines',
                line: {color: '#9a6bff', width: 2},
                fill: 'tozeroy', fillcolor: 'rgba(154, 107, 255, 0.1)'
            }], {
                ...layoutBase, showlegend: false,
                xaxis: {...layoutBase.xaxis, title: 'Disc Radius (mm)'},
                yaxis: {...layoutBase.yaxis, title: 'Reflectivity (%)'}
            }, plotCfg);
        }

        // ---- 4. Thermal Profile ----
        const thData = data.visualizations.thermalProfile;
        Plotly.newPlot('plotlyThermal', [
            { x: thData.timeNs, y: thData.centerTempK, name: 'Center Temp', line: {color: '#f43f5e', width: 2} },
            { x: thData.timeNs, y: thData.edgeTempK, name: 'Edge Temp', line: {color: '#fbbf24', width: 2} }
        ], {
            ...layoutBase, showlegend: true,
            xaxis: {...layoutBase.xaxis, title: 'Time (ns)'},
            yaxis: {...layoutBase.yaxis, title: 'Temperature (K)'}
        }, plotCfg);

        // ---- 5. Manufacturing Quality (Birefringence + Thickness + SNR Profile) ----
        if (mfgData && el('plotlyManufacturing')) {
            const mfgTraces = [
                {
                    x: mfgData.radiusMm, y: mfgData.birefringenceNm,
                    name: 'Warp/Birefringence (nm)', yaxis: 'y1',
                    type: 'scatter', line: {color: '#fbbf24', width: 2}
                },
                {
                    x: mfgData.radiusMm, y: mfgData.thicknessVariancePct,
                    name: 'Thickness Var (%)', yaxis: 'y2',
                    type: 'scatter', line: {color: '#34f5ff', width: 2, dash: 'dot'}
                }
            ];
            if (mfgData.variance_profile) {
                mfgTraces.push({
                    x: mfgData.radiusMm, y: mfgData.variance_profile,
                    name: 'SNR Profile (dB)', yaxis: 'y1',
                    type: 'scatter', line: {color: '#f43f5e', width: 2, dash: 'dash'}
                });
            }
            Plotly.newPlot('plotlyManufacturing', mfgTraces, {
                ...layoutBase, showlegend: true,
                xaxis: {...layoutBase.xaxis, title: 'Disc Radius (mm)'},
                yaxis: {...layoutBase.yaxis, title: 'Birefringence / SNR'},
                yaxis2: {
                    title: 'Thickness Var (%)', overlaying: 'y', side: 'right',
                    gridcolor: 'rgba(255,255,255,0.02)',
                    tickfont: {color: '#34f5ff'}
                }
            }, plotCfg);
        }

    } catch(err) {
        console.error(err);
        alert(err.message);
    } finally {
        btn.textContent = oldText;
        btn.disabled = false;
    }
});
