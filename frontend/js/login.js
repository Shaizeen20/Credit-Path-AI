// Toggle between login and signup forms
function toggleForm(form) {
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');
    const loginBtn = document.querySelectorAll('.toggle-btn')[0];
    const signupBtn = document.querySelectorAll('.toggle-btn')[1];

    if (form === 'login') {
        loginForm.classList.add('active');
        signupForm.classList.remove('active');
        loginBtn.classList.add('active');
        signupBtn.classList.remove('active');
    } else {
        signupForm.classList.add('active');
        loginForm.classList.remove('active');
        signupBtn.classList.add('active');
        loginBtn.classList.remove('active');
    }

    // Clear error messages
    document.querySelectorAll('.error-message').forEach(el => {
        el.classList.remove('show');
    });
}

// Validate email format
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

// Show error message
function showError(elementId, message) {
    const errorEl = document.getElementById(elementId);
    if (errorEl) {
        errorEl.textContent = message;
        errorEl.classList.add('show');
    }
}

// Clear error message
function clearError(elementId) {
    const errorEl = document.getElementById(elementId);
    if (errorEl) {
        errorEl.classList.remove('show');
    }
}

// Login form submission
document.getElementById('loginFormElement').addEventListener('submit', async (e) => {
    e.preventDefault();

    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    const btn = document.getElementById('loginBtn');

    // Clear previous errors
    clearError('loginEmailError');
    clearError('loginPasswordError');

    // Validation
    let hasError = false;
    if (!isValidEmail(email)) {
        showError('loginEmailError', 'Please enter a valid email address');
        hasError = true;
    }
    if (password.length < 6) {
        showError('loginPasswordError', 'Password must be at least 6 characters');
        hasError = true;
    }

    if (hasError) return;

    // API call
    btn.classList.add('loading');
    btn.textContent = 'Logging in...';

    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });

        if (response.ok) {
            const data = await response.json();
            // Store token if provided
            if (data.token) {
                localStorage.setItem('authToken', data.token);
            }
            document.getElementById('loginSuccess').classList.add('show');
            setTimeout(() => {
                window.location.href = '/home';
            }, 1500);
        } else {
            showError('loginPasswordError', 'Invalid email or password');
        }
    } catch (error) {
        showError('loginPasswordError', 'Connection error. Please try again.');
    } finally {
        btn.classList.remove('loading');
        btn.textContent = 'Login to Account';
    }
});

// Signup form submission
document.getElementById('signupFormElement').addEventListener('submit', async (e) => {
    e.preventDefault();

    const firstName = document.getElementById('signupFirstName').value;
    const lastName = document.getElementById('signupLastName').value;
    const company = document.getElementById('signupCompany').value;
    const email = document.getElementById('signupEmail').value;
    const password = document.getElementById('signupPassword').value;
    const confirmPassword = document.getElementById('signupConfirmPassword').value;
    const agreeTerms = document.getElementById('agreeTerms').checked;
    const btn = document.getElementById('signupBtn');

    // Clear previous errors
    document.querySelectorAll('.error-message').forEach(el => el.classList.remove('show'));

    // Validation
    let hasError = false;
    if (firstName.trim().length < 2) {
        showError('signupFirstNameError', 'First name must be at least 2 characters');
        hasError = true;
    }
    if (lastName.trim().length < 2) {
        showError('signupLastNameError', 'Last name must be at least 2 characters');
        hasError = true;
    }
    if (company.trim().length < 2) {
        showError('signupCompanyError', 'Company name is required');
        hasError = true;
    }
    if (!isValidEmail(email)) {
        showError('signupEmailError', 'Please enter a valid email address');
        hasError = true;
    }
    if (password.length < 8) {
        showError('signupPasswordError', 'Password must be at least 8 characters');
        hasError = true;
    }
    if (password !== confirmPassword) {
        showError('signupConfirmPasswordError', 'Passwords do not match');
        hasError = true;
    }
    if (!agreeTerms) {
        showError('signupConfirmPasswordError', 'You must agree to the terms');
        hasError = true;
    }

    if (hasError) return;

    // API call
    btn.classList.add('loading');
    btn.textContent = 'Creating Account...';

    try {
        const response = await fetch('/api/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                firstName,
                lastName,
                company,
                email,
                password
            })
        });

        if (response.ok) {
            document.getElementById('signupSuccess').classList.add('show');
            setTimeout(() => {
                toggleForm('login');
                document.getElementById('signupFormElement').reset();
            }, 1500);
        } else {
            showError('signupConfirmPasswordError', 'This email is already registered');
        }
    } catch (error) {
        showError('signupConfirmPasswordError', 'Connection error. Please try again.');
    } finally {
        btn.classList.remove('loading');
        btn.textContent = 'Create Account';
    }
});

// Real-time error clearing on input
document.querySelectorAll('input').forEach(input => {
    input.addEventListener('input', () => {
        const errorId = input.id + 'Error';
        clearError(errorId);
    });
});