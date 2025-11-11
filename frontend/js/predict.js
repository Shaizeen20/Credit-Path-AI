// API Configuration
const API_BASE_URL = window.location.origin + '/api';

// Form submission handler
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Hide previous results and errors
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('errorMessage').style.display = 'none';
    
    // Show loading spinner
    document.getElementById('loading').style.display = 'block';
    document.querySelector('.btn-predict').disabled = true;

    // Collect and format form data
    const borrowerData = collectFormData();
    
    try {
        // Send prediction request to API
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(borrowerData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || `Server error: ${response.status}`);
        }

        const result = await response.json();
        displayResult(result, borrowerData);
        
    } catch (error) {
        showError(`Unable to process prediction: ${error.message}. Please ensure the API server is running.`);
        console.error('Prediction Error:', error);
    } finally {
        // Hide loading and re-enable button
        document.getElementById('loading').style.display = 'none';
        document.querySelector('.btn-predict').disabled = false;
    }
});

/**
 * Collect and format form data
 */
function collectFormData() {
    const formData = new FormData(document.getElementById('predictionForm'));
    const borrowerData = {};
    
    // Numeric fields
    const numericFields = [
        'person_age', 
        'person_income', 
        'person_emp_exp', 
        'loan_amnt', 
        'loan_int_rate', 
        'cb_person_cred_hist_length', 
        'credit_score'
    ];
    
    formData.forEach((value, key) => {
        if (numericFields.includes(key)) {
            borrowerData[key] = parseFloat(value);
        } else {
            borrowerData[key] = value;
        }
    });
    
    return borrowerData;
}

/**
 * Display prediction results
 */
function displayResult(result, borrowerData) {
    const resultSection = document.getElementById('resultSection');
    const riskBadge = document.getElementById('riskBadge');
    const probabilityFill = document.getElementById('probabilityFill');
    const recommendation = document.getElementById('recommendation');
    const borrowerSummary = document.getElementById('borrowerSummary');

    // Calculate probability percentage
    const probability = Math.round(result.default_probability * 100);
    
    // Determine risk level and styling
    const riskInfo = getRiskInfo(probability);
    
    // Update risk badge
    riskBadge.innerHTML = `<div class="risk-badge ${riskInfo.class}">${riskInfo.level}</div>`;
    
    // Update probability bar with animation
    setTimeout(() => {
        probabilityFill.style.width = `${probability}%`;
        probabilityFill.style.background = riskInfo.color;
        probabilityFill.textContent = `${probability}% Default Probability`;
    }, 100);
    
    // Display borrower summary
    displayBorrowerSummary(borrowerData, result.borrower_summary, borrowerSummary);
    
    // Update recommendation
    updateRecommendation(recommendation, result.recommendation, probability);
    
    // Show result section with smooth scroll
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Get risk information based on probability
 */
function getRiskInfo(probability) {
    if (probability >= 60) {
        return {
            level: 'High Risk',
            class: 'risk-high',
            color: '#ff4757'
        };
    } else if (probability >= 30) {
        return {
            level: 'Medium Risk',
            class: 'risk-medium',
            color: '#ffa502'
        };
    } else {
        return {
            level: 'Low Risk',
            class: 'risk-low',
            color: '#26de81'
        };
    }
}

/**
 * Display borrower summary information
 */
function displayBorrowerSummary(inputData, summaryData, container) {
    const loanToIncomeRatio = summaryData.loan_to_income_ratio || 
                              ((inputData.loan_amnt / inputData.person_income) * 100).toFixed(1);
    
    container.innerHTML = `
        <h3>📋 Borrower Summary</h3>
        <div class="summary-grid">
            <div class="summary-item">
                <label>Annual Income</label>
                <strong>₹${formatCurrency(summaryData.annual_income || inputData.person_income)}</strong>
            </div>
            <div class="summary-item">
                <label>Loan Amount</label>
                <strong>₹${formatCurrency(summaryData.loan_amount || inputData.loan_amnt)}</strong>
            </div>
            <div class="summary-item">
                <label>Credit Score</label>
                <strong>${summaryData.credit_score || inputData.credit_score}</strong>
            </div>
            <div class="summary-item">
                <label>Loan-to-Income Ratio</label>
                <strong>${loanToIncomeRatio}%</strong>
            </div>
            <div class="summary-item">
                <label>Employment Experience</label>
                <strong>${summaryData.employment_experience || inputData.person_emp_exp} years</strong>
            </div>
            <div class="summary-item">
                <label>Credit History</label>
                <strong>${summaryData.credit_history_length || inputData.cb_person_cred_hist_length} years</strong>
            </div>
        </div>
    `;
}

/**
 * Update recommendation section
 */
function updateRecommendation(container, recommendation, probability) {
    let nextStepsList = '';
    if (recommendation.next_steps && Array.isArray(recommendation.next_steps)) {
        nextStepsList = recommendation.next_steps.map(step => `<li>${step}</li>`).join('');
    }
    
    container.innerHTML = `
        <h3>${recommendation.action || 'Action Required'}</h3>
        <p>${recommendation.details || ''}</p>
        
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e0e0e0;">
            <p><strong>Default Probability:</strong> ${probability}%</p>
            <p><strong>Risk Category:</strong> ${recommendation.risk_level || 'Unknown'}</p>
            <p><strong>Priority:</strong> ${recommendation.priority || 'Normal'}</p>
            <p><strong>Timeline:</strong> ${recommendation.timeline || 'Standard'}</p>
        </div>
        
        ${nextStepsList ? `
        <div style="margin-top: 20px;">
            <strong>Recommended Next Steps:</strong>
            <ul style="margin-top: 10px; margin-left: 20px; line-height: 1.8;">
                ${nextStepsList}
            </ul>
        </div>
        ` : ''}
    `;
}

/**
 * Format currency with Indian numbering system
 */
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-IN').format(amount);
}

/**
 * Show error message
 */
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Reset form and hide results
 */
function resetForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('errorMessage').style.display = 'none';
    
    // Reset probability bar
    document.getElementById('probabilityFill').style.width = '0%';
    
    // Smooth scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}