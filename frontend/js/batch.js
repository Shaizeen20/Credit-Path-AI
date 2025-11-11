// API Configuration
const API_BASE_URL = window.location.origin + '/api';

let selectedFile = null;
let resultsData = [];

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeBatchPage();
});

function initializeBatchPage() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    // Click to upload
    uploadArea.addEventListener('click', function(e) {
        if (e.target.tagName !== 'BUTTON') {
            fileInput.click();
        }
    });
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag & Drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFile(files[0]);
        }
    });
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    console.log('File selected:', file.name);
    
    // Validate file type
    if (!file.name.endsWith('.csv')) {
        showError('Please upload a CSV file');
        return;
    }

    selectedFile = file;
    
    // Show file info
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = formatFileSize(file.size);
    document.getElementById('fileInfo').style.display = 'block';
    
    console.log('File info displayed');
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

// Process batch predictions
async function processBatch() {
    console.log('Process batch called');
    
    if (!selectedFile) {
        showError('Please select a CSV file first');
        return;
    }

    // Hide previous results and errors
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('errorMessage').style.display = 'none';
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('processBtn').disabled = true;

    try {
        // Read CSV file
        console.log('Reading CSV file...');
        const csvText = await readFileAsText(selectedFile);
        
        // Parse CSV
        console.log('Parsing CSV...');
        const borrowers = parseCSV(csvText);
        
        if (borrowers.length === 0) {
            throw new Error('No valid data found in CSV file');
        }

        console.log(`Processing ${borrowers.length} borrowers...`);

        // Send batch request to API
        const response = await fetch(`${API_BASE_URL}/predict_batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(borrowers)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.message || `Server error: ${response.status}`);
        }

        const result = await response.json();
        
        // Store results
        resultsData = result.predictions;
        
        console.log('Results received:', resultsData.length);
        
        // Display results
        displayResults(resultsData);
        
    } catch (error) {
        showError(`Failed to process batch: ${error.message}`);
        console.error('Batch processing error:', error);
    } finally {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('processBtn').disabled = false;
    }
}

// Read file as text
function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

// Parse CSV to JSON
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    if (lines.length < 2) {
        throw new Error('CSV file must contain headers and at least one data row');
    }

    // Get headers
    const headers = lines[0].split(',').map(h => h.trim().replace(/['"]/g, ''));
    
    console.log('CSV Headers:', headers);
    
    // Required fields
    const requiredFields = [
        'person_age', 'person_income', 'person_emp_exp', 'person_home_ownership',
        'loan_amnt', 'loan_int_rate', 'loan_intent', 'credit_score',
        'cb_person_cred_hist_length', 'previous_loan_defaults_on_file'
    ];

    // Validate headers
    const missingFields = [];
    for (const field of requiredFields) {
        if (!headers.includes(field)) {
            missingFields.push(field);
        }
    }
    
    if (missingFields.length > 0) {
        throw new Error(`Missing required columns: ${missingFields.join(', ')}`);
    }

    // Parse data rows
    const borrowers = [];
    for (let i = 1; i < lines.length; i++) {
        if (!lines[i].trim()) continue;
        
        const values = lines[i].split(',').map(v => v.trim().replace(/['"]/g, ''));
        const borrower = {};
        
        headers.forEach((header, index) => {
            const value = values[index];
            
            // Convert numeric fields
            if (['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
                 'loan_int_rate', 'credit_score', 'cb_person_cred_hist_length'].includes(header)) {
                borrower[header] = parseFloat(value) || 0;
            } else {
                borrower[header] = value;
            }
        });
        
        borrowers.push(borrower);
    }

    console.log('Parsed borrowers:', borrowers.length);
    return borrowers;
}

// Display results
function displayResults(predictions) {
    console.log('Displaying results...');
    
    // Calculate statistics
    let highRisk = 0, mediumRisk = 0, lowRisk = 0;
    
    predictions.forEach(pred => {
        if (pred.risk_level === 'High Risk') highRisk++;
        else if (pred.risk_level === 'Medium Risk') mediumRisk++;
        else lowRisk++;
    });

    // Update stats
    document.getElementById('totalCount').textContent = predictions.length;
    document.getElementById('highRiskCount').textContent = highRisk;
    document.getElementById('mediumRiskCount').textContent = mediumRisk;
    document.getElementById('lowRiskCount').textContent = lowRisk;

    // Populate table
    const tbody = document.getElementById('resultsBody');
    tbody.innerHTML = '';

    predictions.forEach((pred, index) => {
        const row = document.createElement('tr');
        
        const riskClass = pred.risk_level === 'High Risk' ? 'high' : 
                         pred.risk_level === 'Medium Risk' ? 'medium' : 'low';
        
        const probability = Math.round(pred.default_probability * 100);
        
        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${pred.borrower_summary.age}</td>
            <td>₹${formatCurrency(pred.borrower_summary.annual_income)}</td>
            <td>₹${formatCurrency(pred.borrower_summary.loan_amount)}</td>
            <td>${pred.borrower_summary.credit_score}</td>
            <td>${probability}%</td>
            <td><span class="risk-tag ${riskClass}">${pred.risk_level}</span></td>
            <td>${pred.recommendation.action}</td>
        `;
        
        tbody.appendChild(row);
    });

    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

// Format currency
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-IN').format(amount);
}

// Download results as CSV
function downloadResults() {
    if (resultsData.length === 0) {
        showError('No results to download');
        return;
    }

    // Create CSV content
    let csv = 'Index,Age,Income,Loan Amount,Credit Score,Default Probability,Risk Level,Recommendation,Action Priority\n';
    
    resultsData.forEach((pred, index) => {
        const probability = (pred.default_probability * 100).toFixed(2);
        csv += `${index + 1},`;
        csv += `${pred.borrower_summary.age},`;
        csv += `${pred.borrower_summary.annual_income},`;
        csv += `${pred.borrower_summary.loan_amount},`;
        csv += `${pred.borrower_summary.credit_score},`;
        csv += `${probability}%,`;
        csv += `"${pred.risk_level}",`;
        csv += `"${pred.recommendation.action}",`;
        csv += `"${pred.recommendation.priority}"\n`;
    });

    // Download file
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `creditpath_predictions_${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// Download CSV template
function downloadTemplate() {
    const template = `person_age,person_income,person_emp_exp,person_home_ownership,loan_amnt,loan_int_rate,loan_intent,credit_score,cb_person_cred_hist_length,previous_loan_defaults_on_file
30,500000,5,RENT,150000,10.5,EDUCATION,650,8,No
25,300000,2,OWN,80000,12.0,MEDICAL,580,4,No
35,750000,10,MORTGAGE,200000,9.5,HOMEIMPROVEMENT,720,12,No`;

    const blob = new Blob([template], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'creditpath_template.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// Reset form
function resetForm() {
    selectedFile = null;
    resultsData = [];
    document.getElementById('fileInput').value = '';
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('errorMessage').style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Show error
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    errorDiv.scrollIntoView({ behavior: 'smooth' });
}