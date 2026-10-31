// Decode base64 to Float64Array
function base64ToFloat64Array(b64) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    return new Float64Array(bytes.buffer, bytes.byteOffset, bytes.length / 8);
}

// Cached model references
let modelCache = null;

function getModel() {
    if (!modelCache) {
        if (typeof MODEL_DATA === 'undefined') {
            throw new Error("Model data ('model_data.js') is not loaded.");
        }
        modelCache = {
            numCols: MODEL_DATA.num_cols,
            scalerMean: MODEL_DATA.scaler_mean,
            scalerScale: MODEL_DATA.scaler_scale,
            categories: MODEL_DATA.categories,
            gamma: MODEL_DATA.gamma,
            intercept: MODEL_DATA.intercept,
            nSv: MODEL_DATA.n_sv,
            nFeatures: MODEL_DATA.n_features,
            sv: base64ToFloat64Array(MODEL_DATA.sv_b64),
            dualCoef: base64ToFloat64Array(MODEL_DATA.dual_coef_b64)
        };
    }
    return modelCache;
}

// Predict State of Health (SoH)
function predictSoH(inputs) {
    const model = getModel();

    // 1. Standard Scaling on numerical features
    // Order: ['Battery_Capacity_kWh', 'Vehicle_Age_Months', 'Total_Charging_Cycles', 
    //         'Avg_Temperature_C', 'Fast_Charge_Ratio', 'Avg_Discharge_Rate_C', 'Internal_Resistance_Ohm']
    const numVals = [
        inputs.batteryCapacityKwh,
        inputs.vehicleAgeMonths,
        inputs.totalChargingCycles,
        inputs.avgTemperatureC,
        inputs.fastChargeRatio,
        inputs.avgDischargeRateC,
        inputs.internalResistanceOhm
    ];

    const scaledNum = numVals.map((val, i) => (val - model.scalerMean[i]) / model.scalerScale[i]);

    // 2. One-Hot Encoding for categorical features
    // Order: ['Car_Model', 'Battery_Type', 'Driving_Style']
    const catVals = [inputs.carModel, inputs.batteryType, inputs.drivingStyle];
    const oneHot = [];

    model.categories.forEach((catList, catIdx) => {
        const selected = catVals[catIdx];
        catList.forEach(category => {
            oneHot.push(category === selected ? 1.0 : 0.0);
        });
    });

    const x = scaledNum.concat(oneHot);

    // 3. SVR RBF Kernel: dot(dual_coef, exp(-gamma * ||sv - x||^2)) + intercept
    let dot = 0.0;
    const nSv = model.nSv;
    const nFeat = model.nFeatures;
    const gamma = model.gamma;
    const sv = model.sv;
    const dualCoef = model.dualCoef;

    for (let i = 0; i < nSv; i++) {
        let distSq = 0.0;
        const offset = i * nFeat;
        for (let j = 0; j < nFeat; j++) {
            const diff = sv[offset + j] - x[j];
            distSq += diff * diff;
        }
        dot += dualCoef[i] * Math.exp(-gamma * distSq);
    }

    let soh = dot + model.intercept;
    if (soh >= 0.0 && soh <= 1.0) {
        soh *= 100.0;
    }
    return soh;
}

// UI Event Handlers
document.addEventListener('DOMContentLoaded', () => {
    // Slider live value updates
    const ageSlider = document.getElementById('vehicleAge');
    const ageVal = document.getElementById('vehicleAgeVal');
    ageSlider.addEventListener('input', () => {
        ageVal.textContent = `${ageSlider.value} Months`;
    });

    const fastChargeSlider = document.getElementById('fastCharge');
    const fastChargeVal = document.getElementById('fastChargeVal');
    fastChargeSlider.addEventListener('input', () => {
        fastChargeVal.textContent = parseFloat(fastChargeSlider.value).toFixed(2);
    });

    // Analyze button click
    const analyzeBtn = document.getElementById('analyzeBtn');
    const reportSection = document.getElementById('reportSection');
    const metricVal = document.getElementById('metricVal');
    const diagnosisBox = document.getElementById('diagnosisBox');

    analyzeBtn.addEventListener('click', () => {
        try {
            const payload = {
                carModel: document.getElementById('carModel').value,
                batteryType: document.getElementById('batteryType').value,
                drivingStyle: document.getElementById('drivingStyle').value,
                vehicleAgeMonths: parseFloat(ageSlider.value),
                totalChargingCycles: parseFloat(document.getElementById('chargingCycles').value),
                avgTemperatureC: parseFloat(document.getElementById('temperature').value),
                fastChargeRatio: parseFloat(fastChargeSlider.value),
                avgDischargeRateC: parseFloat(document.getElementById('dischargeSpeed').value),
                internalResistanceOhm: parseFloat(document.getElementById('internalResistance').value),
                batteryCapacityKwh: 75.0 // Hardcoded default
            };

            const soh = predictSoH(payload);

            // Display formatted metric
            metricVal.textContent = `${soh.toFixed(2)}%`;

            // Conditional health diagnosis styling
            diagnosisBox.className = 'alert-box';
            if (soh >= 95.0) {
                diagnosisBox.classList.add('alert-success');
                diagnosisBox.innerHTML = '<strong>System Health Diagnosis</strong>: Optimal Condition Status (Low Degradation Class)';
            } else if (soh >= 85.0) {
                diagnosisBox.classList.add('alert-warning');
                diagnosisBox.innerHTML = '<strong>System Health Diagnosis</strong>: Moderate Capacity Deficit Discovered';
            } else {
                diagnosisBox.classList.add('alert-danger');
                diagnosisBox.innerHTML = '<strong>System Health Diagnosis</strong>: Critical Performance Capacity Alert!';
            }

            reportSection.classList.remove('hidden');
            reportSection.scrollIntoView({ behavior: 'smooth' });

        } catch (err) {
            alert('Prediction error: ' + err.message);
        }
    });
});

