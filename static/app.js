const API_BASE_STORAGE_KEY = "osis.apiBaseUrl";

function normalizeApiBaseUrl(value) {
    let trimmed = String(value ?? "").trim();
    if (!trimmed) return "";
    if (!/^https?:\/\//i.test(trimmed) && /^(localhost|\d{1,3}(?:\.\d{1,3}){3}|[\w.-]+\.[a-z]{2,})(:\d+)?(\/.*)?$/i.test(trimmed)) {
        trimmed = `http://${trimmed}`;
    }
    return trimmed.replace(/\/+$/, "");
}

function resolveInitialApiBaseUrl() {
    const queryBase = normalizeApiBaseUrl(new URLSearchParams(window.location.search).get("api_base"));
    if (queryBase) {
        window.localStorage.setItem(API_BASE_STORAGE_KEY, queryBase);
        return queryBase;
    }

    const globalBase = normalizeApiBaseUrl(window.__OSIS_API_BASE__);
    if (globalBase) return globalBase;

    const storedBase = normalizeApiBaseUrl(window.localStorage.getItem(API_BASE_STORAGE_KEY));
    if (storedBase) return storedBase;

    if (window.location.protocol === "file:" || window.location.port === "5501") {
        return "http://127.0.0.1:8000";
    }

    return "";
}

let apiBaseUrl = resolveInitialApiBaseUrl();

const batchState = {
    enabled: false,
    targetCount: 4,
    batchConfigs: [],
    rankedResults: [],
    running: false,
    helperMessage: ""
};

function getEl(id) {
    return document.getElementById(id);
}

function getApiUrl(path) {
    return `${apiBaseUrl}${path}`;
}

function getActiveApiLabel(baseUrl = apiBaseUrl) {
    return baseUrl || `${window.location.origin} (same origin)`;
}

function updateApiConnectionUi(state, message, displayBase = apiBaseUrl) {
    const input = getEl("api_base_url");
    const badge = getEl("apiStatusBadge");
    const display = getEl("apiBaseDisplay");
    if (!input || !badge || !display) return;

    input.value = displayBase;
    badge.dataset.state = state;
    badge.textContent = message;
    display.innerHTML = `<strong>Active API:</strong> ${escapeHtml(getActiveApiLabel(displayBase))}`;
}

function setApiBaseUrl(nextBaseUrl, persist = true) {
    apiBaseUrl = normalizeApiBaseUrl(nextBaseUrl);

    if (persist) {
        if (apiBaseUrl) {
            window.localStorage.setItem(API_BASE_STORAGE_KEY, apiBaseUrl);
        } else {
            window.localStorage.removeItem(API_BASE_STORAGE_KEY);
        }
    }

    updateApiConnectionUi("checking", "API endpoint updated");
}

async function checkApiConnection(candidateBaseUrl = apiBaseUrl) {
    const targetBaseUrl = normalizeApiBaseUrl(candidateBaseUrl);
    updateApiConnectionUi("checking", "Checking API...", targetBaseUrl);

    try {
        const response = await fetch(`${targetBaseUrl}/api/v1/health`, {
            method: "GET",
            headers: { "Accept": "application/json" }
        });

        if (!response.ok) {
            throw new Error(`Health check failed with status ${response.status}`);
        }

        const payload = await response.json();
        updateApiConnectionUi("connected", `${payload.service || "OSIS API"} connected`, targetBaseUrl);
        return true;
    } catch (error) {
        console.error(error);
        updateApiConnectionUi("error", "API unreachable", targetBaseUrl);
        return false;
    }
}

function formatFixed(value, digits = 3, suffix = "") {
    const num = Number(value);
    if (!Number.isFinite(num)) return "--";
    return `${num.toFixed(digits)}${suffix}`;
}

function formatExponential(value, digits = 3) {
    const num = Number(value);
    if (!Number.isFinite(num)) return "--";
    return num.toExponential(digits);
}

function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function scrollToElement(element) {
    if (!element) return;

    if (typeof lenis !== "undefined") {
        lenis.scrollTo(element, { offset: -100 });
    } else {
        element.scrollIntoView({ behavior: "smooth" });
    }
}

function calculateOptimizationScore(hybridSnrDb, postFecBer) {
    const snr = Number(hybridSnrDb) || 0;
    const boundedBer = Math.max(Number(postFecBer) || 0, 1e-15);
    return snr - (10 * Math.log10(boundedBer));
}

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
    const response = await fetch(getApiUrl(url), {
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
    const chartEl = getEl("snrChart");
    if (!chartEl) return;

    if (snrChart) {
        snrChart.destroy();
        snrChart = null;
    }

    if (!Array.isArray(simFrames) || simFrames.length === 0) return;

    const labels = simFrames.map(f => Number(f.value).toFixed(3));
    const snrValues = simFrames.map(f => Number(f.predicted_snr_db).toFixed(3));
    const ctx = chartEl.getContext("2d");

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
    const chartEl = getEl("simulationChart");
    if (!chartEl) return;

    if (simulationChart) {
        simulationChart.destroy();
        simulationChart = null;
    }

    if (!Array.isArray(simFrames) || simFrames.length === 0) return;

    const labels = simFrames.map(f => Number(f.value).toFixed(3));
    const snr = simFrames.map(f => f.predicted_snr_db);
    const ber = simFrames.map(f => f.estimated_ber);
    const ctx = chartEl.getContext("2d");

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
    const body = getEl("optimizationBody");
    body.innerHTML = "";
    rows.forEach((row, idx) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${idx + 1}</td>
            <td>${formatFixed(row.numerical_aperture, 3)}</td>
            <td>${formatFixed(row.track_pitch_nm, 2)}</td>
            <td>${formatFixed(row.temperature_c, 1)}</td>
            <td>${formatFixed(row.relative_humidity, 1)}</td>
            <td>${formatFixed(row.predicted_snr_db, 3)}</td>
            <td>${formatExponential(row.estimated_ber, 3)}</td>
        `;
        body.appendChild(tr);
    });
}

function renderSensitivityList(items) {
    const container = getEl("sensitivityBody");
    container.innerHTML = "";
    items.slice(0, 7).forEach(item => {
        const div = document.createElement("div");
        div.className = "sensitivity-item";
        div.innerHTML = `
            <span>${escapeHtml(item.parameter)}</span>
            <span>${formatFixed(item.normalized_sensitivity, 4)}</span>
        `;
        container.appendChild(div);
    });
}

function updateMetricCards(snrData, berData, comparisonData) {
    getEl("physicsSnrValue").textContent = `${formatFixed(snrData.physics_snr_db, 3)} dB`;

    const lower = Number(snrData.snr_lower_bound_db);
    const upper = Number(snrData.snr_upper_bound_db);
    if (Number.isFinite(lower) && Number.isFinite(upper)) {
        getEl("hybridSnrValue").innerHTML = `
          ${formatFixed(snrData.predicted_snr_db, 3)} dB
          <div style="font-size: 0.8rem; color: #9a6bff; margin-top: 4px;">
            90% CI: [${lower.toFixed(2)}, ${upper.toFixed(2)}]
          </div>
        `;
    } else {
        getEl("hybridSnrValue").textContent = `${formatFixed(snrData.predicted_snr_db, 3)} dB`;
    }

    getEl("berValue").textContent = formatExponential(berData.estimated_ber, 3);
    getEl("gainValue").textContent = `${formatFixed(comparisonData.snr_gain_over_analytical_db, 3)} dB`;
}

function renderShap(shapData) {
    const container = getEl("shapBody");
    if(!container) return;
    container.innerHTML = "";
    shapData.forEach(item => {
        const div = document.createElement("div");
        div.className = "sensitivity-item";
        
        const impactColor = item.impact > 0 ? '#34d399' : '#f43f5e';
        const sign = item.impact > 0 ? '+' : '';
        
        div.innerHTML = `
            <span>${escapeHtml(item.feature)}</span>
            <span style="color: ${impactColor}; font-weight: 600;">${sign}${formatFixed(item.impact, 4)} dB</span>
        `;
        container.appendChild(div);
    });
}

function renderComparisonText(cmpData) {
    const hasMeasured = cmpData.measured_snr_db !== undefined;
    const measuredLine = hasMeasured
        ? `Measured SNR: ${formatFixed(cmpData.measured_snr_db, 3)} dB | Analytical Error: ${formatFixed(cmpData.abs_error_analytical_db, 3)} dB | Hybrid Error: ${formatFixed(cmpData.abs_error_ml_hybrid_db, 3)} dB`
        : "Measured SNR not provided.";

    getEl("comparisonText").textContent =
        `Analytical SNR: ${formatFixed(cmpData.analytical_physics_snr_db, 3)} dB | Hybrid SNR: ${formatFixed(cmpData.ml_hybrid_snr_db, 3)} dB | BER Reduction Ratio: ${formatFixed(cmpData.ber_reduction_ratio, 3)} | ${measuredLine}`;
}

function getPrimarySubmitButton() {
    return document.querySelector("#osisForm .submit-btn");
}

function clampBatchCount(value) {
    const numeric = parseInt(value, 10);
    if (!Number.isFinite(numeric)) return 4;
    return Math.min(12, Math.max(2, numeric));
}

function generateBatchClientId() {
    return `osis-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function buildBatchSnapshot() {
    const config = buildPayload();
    const batchIndex = batchState.batchConfigs.length + 1;

    return {
        clientId: generateBatchClientId(),
        label: `Simulation ${batchIndex}`,
        modulation: getEl("modulation").value,
        config,
        reportInputs: {
            laser_wavelength_nm: config.laser_wavelength_nm,
            numerical_aperture: config.numerical_aperture,
            track_pitch_nm: config.track_pitch_nm,
            temperature_c: config.temperature_c,
            relative_humidity: config.relative_humidity,
            recording_material: config.recording_material,
            sputtering_rate_nm_s: parseFloat(getEl("sputtering_rate").value),
            base_thickness_nm: parseFloat(getEl("base_thickness").value),
            simulation_mode: getEl("simulation_mode").value
        }
    };
}

function clearBatchResults() {
    batchState.rankedResults = [];
    getEl("batchResult").style.display = "none";
    getEl("batchResultSummary").textContent = "Ranked batch simulations will appear here after the final configuration is submitted.";
    getEl("batchSizeValue").textContent = "--";
    getEl("batchBestSnrValue").textContent = "-- dB";
    getEl("batchLowestBerValue").textContent = "--";
    getEl("batchSpreadValue").textContent = "--";
    getEl("batchWinnerPanel").innerHTML = "";
    getEl("batchRankingList").innerHTML = "";
    getEl("batchSummaryBody").innerHTML = "";
    getEl("downloadBatchPdfBtn").disabled = true;
}

function renderBatchQueue() {
    const container = getEl("batchConfigList");
    container.innerHTML = "";

    if (!batchState.enabled) {
        container.innerHTML = `
            <div class="batch-chip">
                <strong>Single-run ready</strong>
                <small>The current prediction pipeline stays untouched until batch mode is enabled.</small>
            </div>
        `;
        return;
    }

    if (!batchState.batchConfigs.length) {
        container.innerHTML = `
            <div class="batch-chip">
                <strong>No saved configurations yet</strong>
                <small>Submit the form to store Simulation 1 in the temporary batchConfigs queue.</small>
            </div>
        `;
        return;
    }

    batchState.batchConfigs.forEach((snapshot) => {
        const chip = document.createElement("div");
        chip.className = "batch-chip";
        chip.innerHTML = `
            <strong>${escapeHtml(snapshot.label)}</strong>
            <small>NA ${formatFixed(snapshot.reportInputs.numerical_aperture, 3)} | Temp ${formatFixed(snapshot.reportInputs.temperature_c, 1)} C | Sputter ${formatFixed(snapshot.reportInputs.sputtering_rate_nm_s, 1)} nm/s</small>
        `;
        container.appendChild(chip);
    });
}

function updateSubmitButtonLabel() {
    const button = getPrimarySubmitButton();
    if (!button) return;

    if (!button.dataset.singleLabel) {
        button.dataset.singleLabel = button.textContent.trim();
    }

    if (batchState.running) return;

    if (!batchState.enabled) {
        button.textContent = button.dataset.singleLabel;
        return;
    }

    const nextStep = Math.min(batchState.batchConfigs.length + 1, batchState.targetCount);
    const isFinalCapture = batchState.batchConfigs.length === batchState.targetCount - 1;
    button.textContent = isFinalCapture
        ? `Save Configuration ${nextStep} of ${batchState.targetCount} and Run Batch Ranking`
        : `Save Configuration ${nextStep} of ${batchState.targetCount}`;
}

function updateBatchUi() {
    batchState.targetCount = clampBatchCount(getEl("batch_count").value);
    getEl("batch_count").value = String(batchState.targetCount);

    const titleEl = getEl("batchWizardTitle");
    const textEl = getEl("batchWizardText");
    const badgeEl = getEl("batchStepBadge");
    const progressEl = getEl("batchProgressBar");
    const noteEl = getEl("batchInlineNote");
    const resetBtn = getEl("resetBatchBtn");

    if (!batchState.enabled) {
        titleEl.textContent = "Single-run mode is active";
        textEl.textContent = "Enable batch mode to queue multiple OSIS configurations before the ML model and ranking engine start.";
        badgeEl.textContent = "1 / 1";
        progressEl.style.width = "0%";
        noteEl.textContent = "Batch ranking uses Hybrid SNR and Post-FEC BER after all configurations are collected.";
        resetBtn.disabled = true;
        renderBatchQueue();
        updateSubmitButtonLabel();
        return;
    }

    const capturedCount = batchState.batchConfigs.length;
    const nextStep = Math.min(capturedCount + 1, batchState.targetCount);
    const progressPct = Math.min((capturedCount / batchState.targetCount) * 100, 100);

    if (batchState.running) {
        titleEl.textContent = "Running batch inference and ranking";
        textEl.textContent = "All saved configurations are now passing through the existing ML predictor asynchronously.";
        badgeEl.textContent = `${batchState.targetCount} / ${batchState.targetCount}`;
    } else if (batchState.rankedResults.length) {
        titleEl.textContent = "Batch ranking complete";
        textEl.textContent = batchState.helperMessage || "The ranked results are ready below.";
        badgeEl.textContent = `${batchState.targetCount} / ${batchState.targetCount}`;
    } else {
        titleEl.textContent = `Configuration ${nextStep} of ${batchState.targetCount}`;
        textEl.textContent = batchState.helperMessage || `Submit the current optical parameter set to store it in batchConfigs. The ML engine will not run until configuration ${batchState.targetCount} is submitted.`;
        badgeEl.textContent = `${nextStep} / ${batchState.targetCount}`;
    }

    progressEl.style.width = `${progressPct}%`;
    noteEl.textContent = `Temporary queue: ${capturedCount} of ${batchState.targetCount} configurations captured in batchConfigs.`;
    resetBtn.disabled = capturedCount === 0 && batchState.rankedResults.length === 0;

    renderBatchQueue();
    updateSubmitButtonLabel();
}

function buildOptimizationConclusion(rankedResults) {
    if (!rankedResults.length) return "";

    const winner = rankedResults[0];
    const runnerUp = rankedResults[1];
    const input = winner.reportInputs || winner.inputConfig || {};
    const snrLead = runnerUp ? Number(winner.hybrid_snr_db) - Number(runnerUp.hybrid_snr_db) : 0;
    const berRatio = runnerUp
        ? Number(runnerUp.post_fec_ber) / Math.max(Number(winner.post_fec_ber), 1e-15)
        : null;
    const parameterLine = `NA ${formatFixed(input.numerical_aperture, 3)}, track pitch ${formatFixed(input.track_pitch_nm, 0)} nm, temperature ${formatFixed(input.temperature_c, 1)} C, humidity ${formatFixed(input.relative_humidity, 1)}%, and sputtering rate ${formatFixed(input.sputtering_rate_nm_s, 1)} nm/s`;

    if (!runnerUp) {
        return `${winner.label} is the only configuration in the batch, and ${parameterLine} produced the strongest available Hybrid SNR / Post-FEC BER tradeoff.`;
    }

    const berLine = berRatio && Number.isFinite(berRatio)
        ? `It also improved BER by ${berRatio.toFixed(2)}x compared with Rank 2.`
        : "It also preserved the lowest Post-FEC BER in the batch.";

    return `${winner.label} won because ${parameterLine} produced the highest combined optimization score. It achieved ${formatFixed(winner.hybrid_snr_db, 3)} dB Hybrid SNR with Post-FEC BER ${formatExponential(winner.post_fec_ber, 3)}, leading Rank 2 by ${snrLead.toFixed(3)} dB. ${berLine}`;
}

function renderBatchResults(rankedResults, summary) {
    if (!rankedResults.length) return;

    const winner = rankedResults[0];
    const runnerUp = rankedResults[1];
    const scoreSpread = Number(summary?.score_spread ?? (winner.optimization_score - rankedResults[rankedResults.length - 1].optimization_score));
    const winnerInput = winner.reportInputs || winner.inputConfig || {};
    const conclusion = buildOptimizationConclusion(rankedResults);
    const snrLeadText = runnerUp
        ? `Hybrid SNR lead vs Rank 2: +${(Number(winner.hybrid_snr_db) - Number(runnerUp.hybrid_snr_db)).toFixed(3)} dB`
        : "Only configuration in the batch";

    getEl("batchResultSummary").textContent = `${rankedResults.length} configurations were ranked. Rank 1 is ${winner.label} with the strongest Hybrid SNR / Post-FEC BER balance.`;
    getEl("batchSizeValue").textContent = String(rankedResults.length);
    getEl("batchBestSnrValue").textContent = `${formatFixed(summary?.best_hybrid_snr_db ?? winner.hybrid_snr_db, 3)} dB`;
    getEl("batchLowestBerValue").textContent = formatExponential(summary?.lowest_post_fec_ber ?? winner.post_fec_ber, 3);
    getEl("batchSpreadValue").textContent = formatFixed(scoreSpread, 3);

    getEl("batchWinnerPanel").innerHTML = `
        <div class="winner-header">
            <div>
                <div class="winner-label">Rank 1 Optimal Configuration</div>
                <h3>${escapeHtml(winner.label)}</h3>
            </div>
            <span class="rank-badge rank-badge--winner">Rank ${winner.rank}</span>
        </div>
        <div class="winner-metrics">
            <div class="winner-metric winner-stat">
                <span>Hybrid SNR</span>
                <strong>${formatFixed(winner.hybrid_snr_db, 3)} dB</strong>
            </div>
            <div class="winner-metric winner-stat">
                <span>Post-FEC BER</span>
                <strong>${formatExponential(winner.post_fec_ber, 3)}</strong>
            </div>
            <div class="winner-metric winner-stat">
                <span>Optimization Score</span>
                <strong>${formatFixed(winner.optimization_score ?? calculateOptimizationScore(winner.hybrid_snr_db, winner.post_fec_ber), 3)}</strong>
            </div>
        </div>
        <div class="winner-parameter-grid">
            <div class="winner-parameter"><span>Laser Wavelength</span><strong>${formatFixed(winnerInput.laser_wavelength_nm, 0)} nm</strong></div>
            <div class="winner-parameter"><span>Numerical Aperture</span><strong>${formatFixed(winnerInput.numerical_aperture, 3)}</strong></div>
            <div class="winner-parameter"><span>Track Pitch</span><strong>${formatFixed(winnerInput.track_pitch_nm, 0)} nm</strong></div>
            <div class="winner-parameter"><span>Temperature</span><strong>${formatFixed(winnerInput.temperature_c, 1)} C</strong></div>
            <div class="winner-parameter"><span>Humidity</span><strong>${formatFixed(winnerInput.relative_humidity, 1)}%</strong></div>
            <div class="winner-parameter"><span>Sputtering Rate</span><strong>${formatFixed(winnerInput.sputtering_rate_nm_s, 1)} nm/s</strong></div>
        </div>
        <div class="winner-gain">${snrLeadText}</div>
        <div class="conclusion-block" id="batchConclusionText">${escapeHtml(conclusion)}</div>
    `;

    getEl("batchRankingList").innerHTML = rankedResults.map((result) => {
        const input = result.reportInputs || result.inputConfig || {};
        return `
            <article class="ranked-result-card ${result.rank === 1 ? "is-winner" : ""}">
                <div class="ranked-card-head">
                    <div>
                        <h4>${escapeHtml(result.label)}</h4>
                        <div class="table-muted">Material: ${escapeHtml(input.recording_material || "--")} | Modulation: ${escapeHtml(result.modulation || "--")}</div>
                    </div>
                    <span class="rank-badge ${result.rank === 1 ? "rank-badge--winner" : ""}">Rank ${result.rank}</span>
                </div>
                <div class="ranked-card-metrics">
                    <div class="ranked-card-metric">
                        <span>Hybrid SNR</span>
                        <strong>${formatFixed(result.hybrid_snr_db, 3)} dB</strong>
                    </div>
                    <div class="ranked-card-metric">
                        <span>Post-FEC BER</span>
                        <strong>${formatExponential(result.post_fec_ber, 3)}</strong>
                    </div>
                    <div class="ranked-card-metric">
                        <span>Score</span>
                        <strong>${formatFixed(result.optimization_score ?? calculateOptimizationScore(result.hybrid_snr_db, result.post_fec_ber), 3)}</strong>
                    </div>
                </div>
                <div class="batch-summary-text">
                    Wavelength ${formatFixed(input.laser_wavelength_nm, 0)} nm | NA ${formatFixed(input.numerical_aperture, 3)} | Track ${formatFixed(input.track_pitch_nm, 0)} nm | Temp ${formatFixed(input.temperature_c, 1)} C | Humidity ${formatFixed(input.relative_humidity, 1)}% | Sputtering ${formatFixed(input.sputtering_rate_nm_s, 1)} nm/s
                </div>
            </article>
        `;
    }).join("");

    getEl("batchSummaryBody").innerHTML = rankedResults.map((result) => {
        const input = result.reportInputs || result.inputConfig || {};
        return `
            <tr class="${result.rank === 1 ? "table-highlight" : ""}">
                <td>${result.rank}</td>
                <td>${escapeHtml(result.label)}</td>
                <td>${formatFixed(input.numerical_aperture, 3)}</td>
                <td>${formatFixed(input.track_pitch_nm, 0)} nm</td>
                <td>${formatFixed(input.temperature_c, 1)} C</td>
                <td>${formatFixed(input.relative_humidity, 1)}%</td>
                <td>${formatFixed(input.sputtering_rate_nm_s, 1)} nm/s</td>
                <td>${formatFixed(result.hybrid_snr_db, 3)} dB</td>
                <td>${formatExponential(result.post_fec_ber, 3)}</td>
                <td>${formatFixed(result.optimization_score ?? calculateOptimizationScore(result.hybrid_snr_db, result.post_fec_ber), 3)}</td>
            </tr>
        `;
    }).join("");

    getEl("batchResult").style.display = "block";
    getEl("downloadBatchPdfBtn").disabled = false;
}

function animateBatchResults() {
    if (typeof gsap === "undefined") return;

    const panel = getEl("batchResult");
    const header = panel.querySelector(".batch-result-header");
    const overviewCards = panel.querySelectorAll(".batch-overview-grid .metric-card");
    const winnerPanel = panel.querySelector(".winner-panel");
    const winnerStats = panel.querySelectorAll(".winner-stat");
    const winnerParameters = panel.querySelectorAll(".winner-parameter");
    const winnerGain = panel.querySelector(".winner-gain");
    const conclusion = panel.querySelector(".conclusion-block");
    const rankedCards = panel.querySelectorAll(".ranked-result-card");

    const timeline = gsap.timeline({ defaults: { ease: "power3.out" } });
    timeline
        .from(header, { opacity: 0, y: 20, duration: 0.4 })
        .from(overviewCards, { opacity: 0, y: 18, stagger: 0.08, duration: 0.32 }, "-=0.18")
        .fromTo(winnerPanel, { opacity: 0, y: 26, scale: 0.96 }, { opacity: 1, y: 0, scale: 1, duration: 0.5 }, "-=0.08")
        .from(winnerStats, { opacity: 0, y: 14, stagger: 0.06, duration: 0.24 }, "-=0.28")
        .from(winnerParameters, { opacity: 0, x: -18, stagger: 0.08, duration: 0.26 }, "-=0.18")
        .from(winnerGain, { opacity: 0, scale: 0.82, duration: 0.28 }, "-=0.08")
        .from(conclusion, { opacity: 0, x: 18, duration: 0.28 }, "-=0.16")
        .from(rankedCards, { opacity: 0, y: 20, stagger: 0.1, duration: 0.34 }, "-=0.1");
}

function downloadBatchReport() {
    if (!batchState.rankedResults.length) {
        alert("Run a batch simulation before downloading the report.");
        return;
    }

    const jsPDFCtor = window.jspdf?.jsPDF;
    if (!jsPDFCtor) {
        alert("jsPDF is unavailable in this browser session.");
        return;
    }

    const doc = new jsPDFCtor({ orientation: "landscape", unit: "pt", format: "a4" });
    if (typeof doc.autoTable !== "function") {
        alert("jsPDF-AutoTable is unavailable in this browser session.");
        return;
    }

    const pageWidth = doc.internal.pageSize.getWidth();
    const winner = batchState.rankedResults[0];
    const runnerUp = batchState.rankedResults[1];
    const conclusion = buildOptimizationConclusion(batchState.rankedResults);
    const generatedAt = new Date().toLocaleString();

    const rows = batchState.rankedResults.map((result) => {
        const input = result.reportInputs || result.inputConfig || {};
        return [
            result.rank,
            result.label,
            formatFixed(input.laser_wavelength_nm, 0),
            formatFixed(input.numerical_aperture, 3),
            formatFixed(input.track_pitch_nm, 0),
            formatFixed(input.temperature_c, 1),
            formatFixed(input.relative_humidity, 1),
            formatFixed(input.sputtering_rate_nm_s, 1),
            formatFixed(result.hybrid_snr_db, 3),
            formatExponential(result.post_fec_ber, 3),
            formatFixed(result.optimization_score ?? calculateOptimizationScore(result.hybrid_snr_db, result.post_fec_ber), 3)
        ];
    });

    doc.setFillColor(7, 10, 26);
    doc.rect(0, 0, pageWidth, 88, "F");
    doc.setTextColor(238, 242, 255);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(22);
    doc.text("OSIS Batch Simulation Report", 40, 42);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(185, 194, 234);
    doc.text(`Generated: ${generatedAt}`, 40, 62);
    doc.text("Ranking objective: maximize Hybrid SNR and minimize Post-FEC BER", 40, 78);

    doc.setTextColor(31, 41, 55);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(13);
    doc.text("Optimal Configuration Summary", 40, 118);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(11);
    doc.text(
        `Rank 1: ${winner.label} | Hybrid SNR ${formatFixed(winner.hybrid_snr_db, 3)} dB | Post-FEC BER ${formatExponential(winner.post_fec_ber, 3)} | Score ${formatFixed(winner.optimization_score, 3)}`,
        40,
        138
    );

    const conclusionLines = doc.splitTextToSize(conclusion, pageWidth - 110);
    const conclusionBoxHeight = 42 + (conclusionLines.length * 14);
    doc.setFillColor(245, 248, 255);
    doc.setDrawColor(210, 220, 236);
    doc.roundedRect(36, 154, pageWidth - 72, conclusionBoxHeight, 12, 12, "FD");
    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor(10, 16, 48);
    doc.text("Optimization Conclusion", 52, 178);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(55, 65, 81);
    doc.text(conclusionLines, 52, 196);

    doc.autoTable({
        startY: 154 + conclusionBoxHeight + 18,
        head: [[
            "Rank",
            "Config",
            "Wavelength (nm)",
            "NA",
            "Track Pitch (nm)",
            "Temp (C)",
            "Humidity (%)",
            "Sputtering (nm/s)",
            "Hybrid SNR (dB)",
            "Post-FEC BER",
            "Score"
        ]],
        body: rows,
        theme: "grid",
        styles: {
            fontSize: 9,
            cellPadding: 6,
            textColor: [31, 41, 55],
            lineColor: [214, 223, 239]
        },
        headStyles: {
            fillColor: [10, 16, 48],
            textColor: [238, 242, 255],
            fontStyle: "bold"
        },
        alternateRowStyles: {
            fillColor: [247, 250, 255]
        },
        didParseCell(data) {
            if (data.section === "body" && batchState.rankedResults[data.row.index]?.rank === 1) {
                data.cell.styles.fillColor = [255, 248, 220];
                data.cell.styles.textColor = [31, 41, 55];
                data.cell.styles.fontStyle = "bold";
            }
        }
    });

    const footerY = doc.lastAutoTable.finalY + 20;
    const runnerUpText = runnerUp
        ? `Rank 1 led Rank 2 by ${(Number(winner.hybrid_snr_db) - Number(runnerUp.hybrid_snr_db)).toFixed(3)} dB in Hybrid SNR.`
        : "Only one configuration was included in this batch.";
    doc.setFontSize(9);
    doc.setTextColor(75, 85, 99);
    doc.text("Optimization score = Hybrid SNR - 10*log10(Post-FEC BER).", 40, footerY);
    doc.text(runnerUpText, 40, footerY + 14);

    const dateStamp = new Date().toISOString().slice(0, 10);
    doc.save(`osis-batch-simulation-report-${dateStamp}.pdf`);
}

function resetBatchWorkflow(keepMode) {
    batchState.batchConfigs = [];
    batchState.rankedResults = [];
    batchState.running = false;
    batchState.enabled = keepMode;
    batchState.helperMessage = batchState.enabled ? "Batch queue cleared. Start again with configuration 1." : "";
    clearBatchResults();
    updateBatchUi();
}

async function runSingleAnalysis() {
    const btn = getPrimarySubmitButton();
    const originalText = btn.dataset.singleLabel || btn.textContent;
    btn.textContent = "Running analysis...";
    btn.disabled = true;

    try {
        const basePayload = buildPayload();
        const modulation = getEl("modulation").value;
        const measuredInput = getEl("measured_snr").value.trim();
        const topK = parseInt(getEl("top_k").value, 10) || 3;
        const deltaFraction = parseFloat(getEl("delta_fraction").value) || 0.05;
        const sweepParameter = getEl("sweep_param").value;
        const simStart = parseFloat(getEl("sim_start").value);
        const simEnd = parseFloat(getEl("sim_end").value);
        const simSteps = parseInt(getEl("sim_steps").value, 10) || 20;

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

        const simMeta = getEl("simMeta");
        if (simMeta) {
            simMeta.textContent = `${frames.length} frames generated for ${sweepParameter} sweep from ${simStart} to ${simEnd}.`;
        }

        const resultDiv = getEl("result");
        resultDiv.style.display = "block";
        scrollToElement(resultDiv);
    } catch (error) {
        console.error(error);
        alert("Analysis failed. Ensure backend is running and inputs are valid.");
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

async function runBatchInference() {
    const btn = getPrimarySubmitButton();
    batchState.running = true;
    batchState.helperMessage = "All configurations captured. Running asynchronous batch inference and optimization ranking.";
    updateBatchUi();
    btn.textContent = "Ranking batch configurations...";
    btn.disabled = true;

    try {
        const response = await postJson("/api/v1/batch_simulations", {
            modulation: getEl("modulation").value,
            batchConfigs: batchState.batchConfigs.map((snapshot) => ({
                clientId: snapshot.clientId,
                label: snapshot.label,
                modulation: snapshot.modulation,
                config: snapshot.config
            }))
        });

        const snapshotMap = new Map(batchState.batchConfigs.map((snapshot) => [snapshot.clientId, snapshot]));
        batchState.rankedResults = (response.rankedResults || []).map((result) => {
            const snapshot = snapshotMap.get(result.clientId);
            return {
                ...result,
                label: snapshot?.label || result.label,
                reportInputs: snapshot?.reportInputs || result.inputConfig || {}
            };
        });

        batchState.helperMessage = `Batch ranking complete. ${batchState.rankedResults[0]?.label || "Rank 1"} is highlighted as the optimal configuration.`;
        renderBatchResults(batchState.rankedResults, response.summary || {});
        animateBatchResults();
        scrollToElement(getEl("batchResult"));
    } catch (error) {
        console.error(error);
        batchState.helperMessage = "Batch inference failed. Review the queued configurations and try again.";
        alert("Batch simulation failed. Ensure backend is running and all saved configurations are valid.");
    } finally {
        batchState.running = false;
        btn.disabled = false;
        updateBatchUi();
    }
}

async function handleBatchSubmit() {
    const btn = getPrimarySubmitButton();
    btn.disabled = true;

    try {
        const snapshot = buildBatchSnapshot();
        batchState.batchConfigs = [...batchState.batchConfigs, snapshot];
        clearBatchResults();

        if (batchState.batchConfigs.length < batchState.targetCount) {
            const nextStep = batchState.batchConfigs.length + 1;
            batchState.helperMessage = `${snapshot.label} stored. Update the form for configuration ${nextStep} of ${batchState.targetCount} and submit again.`;
            updateBatchUi();
            return;
        }

        await runBatchInference();
    } finally {
        if (!batchState.running) {
            btn.disabled = false;
            updateBatchUi();
        }
    }
}

const triggerInputs = ["wavelength", "na", "track_pitch", "layer_spacing"];
triggerInputs.forEach(id => {
    getEl(id).addEventListener("input", () => {
        const formatSelect = getEl("disc_format");
        if (formatSelect) formatSelect.value = "custom";
        calculatePhysics();
    });
});

const formatEl = getEl("disc_format");
if (formatEl) {
    formatEl.addEventListener("change", (e) => {
        const format = e.target.value;
        if (format === "cd") {
            getEl("wavelength").value = "780";
            getEl("na").value = "0.45";
            getEl("track_pitch").value = "1600";
            getEl("layer_spacing").value = "0";
            getEl("layer_count").value = "1";
        } else if (format === "dvd") {
            getEl("wavelength").value = "650";
            getEl("na").value = "0.60";
            getEl("track_pitch").value = "740";
            getEl("layer_spacing").value = "55000";
            getEl("layer_count").value = "2";
        } else if (format === "bd") {
            getEl("wavelength").value = "405";
            getEl("na").value = "0.85";
            getEl("track_pitch").value = "320";
            getEl("layer_spacing").value = "25000";
            getEl("layer_count").value = "2";
        }
        calculatePhysics();
    });
}
getEl("sweep_param").addEventListener("change", applySweepDefaults);

calculatePhysics();
applySweepDefaults();
clearBatchResults();
updateBatchUi();
updateApiConnectionUi("checking", "Resolving API...");

getEl("batch_mode").addEventListener("change", (e) => {
    resetBatchWorkflow(e.target.checked);
});

getEl("saveApiBaseBtn").addEventListener("click", async () => {
    setApiBaseUrl(getEl("api_base_url").value);
    const connected = await checkApiConnection();
    if (!connected) {
        alert("The API base URL was saved, but the backend did not respond to the health check.");
    }
});

getEl("testApiConnectionBtn").addEventListener("click", async () => {
    const connected = await checkApiConnection(getEl("api_base_url").value);
    if (!connected) {
        alert("Unable to reach the OSIS API with the current endpoint.");
        updateApiConnectionUi("error", "API unreachable");
    }
});

getEl("batch_count").addEventListener("change", () => {
    const nextCount = clampBatchCount(getEl("batch_count").value);
    const countChanged = nextCount !== batchState.targetCount;
    batchState.targetCount = nextCount;
    getEl("batch_count").value = String(nextCount);

    if (countChanged && (batchState.batchConfigs.length || batchState.rankedResults.length)) {
        batchState.batchConfigs = [];
        batchState.helperMessage = "Batch size changed. The temporary queue was cleared to avoid mixing steps.";
        clearBatchResults();
    }

    updateBatchUi();
});

getEl("resetBatchBtn").addEventListener("click", () => {
    resetBatchWorkflow(batchState.enabled);
});

getEl("downloadBatchPdfBtn").addEventListener("click", downloadBatchReport);

getEl("osisForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    if (batchState.enabled) {
        await handleBatchSubmit();
        return;
    }

    await runSingleAnalysis();
});

checkApiConnection();

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

        const res = await fetch(getApiUrl('/api/v1/simulate_platform'), {
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
