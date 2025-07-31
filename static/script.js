// SEOExtract JavaScript functionality

// Copy keywords functionality
function copyKeywords() {
    const keywordsList = document.getElementById('keywordsList');
    if (!keywordsList) {
        console.error('Keywords list not found');
        return;
    }

    const keywords = keywordsList.querySelectorAll('.badge');
    const keywordText = Array.from(keywords).map(badge => badge.textContent.trim()).join('\n');

    if (navigator.clipboard) {
        navigator.clipboard.writeText(keywordText).then(() => {
            // Show success message
            const btn = event.target.closest('button');
            if (btn) {
                const originalText = btn.innerHTML;
                btn.innerHTML = '<i class="fas fa-check me-1"></i>Copied!';
                btn.classList.add('btn-success');
                btn.classList.remove('btn-outline-primary');

                setTimeout(() => {
                    btn.innerHTML = originalText;
                    btn.classList.remove('btn-success');
                    btn.classList.add('btn-outline-primary');
                }, 2000);
            }
        }).catch(err => {
            console.error('Failed to copy: ', err);
            alert('Failed to copy keywords. Please select and copy manually.');
        });
    } else {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = keywordText;
        document.body.appendChild(textArea);
        textArea.select();
        try {
            document.execCommand('copy');
            alert('Keywords copied to clipboard!');
        } catch (err) {
            console.error('Fallback copy failed: ', err);
            alert('Failed to copy keywords. Please select and copy manually.');
        }
        document.body.removeChild(textArea);
    }
}

// Fallback copy method for older browsers
function fallbackCopy(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();

    try {
        document.execCommand('copy');
        showToast('Keywords copied to clipboard!', 'success');
    } catch (err) {
        console.error('Fallback copy failed: ', err);
        showToast('Copy failed. Please copy manually.', 'error');
    }

    document.body.removeChild(textArea);
}

document.addEventListener('DOMContentLoaded', function() {
    // Quick demo form handling
    const quickDemo = document.getElementById('quickDemo');
    if (quickDemo) {
        quickDemo.addEventListener('submit', function(e) {
            e.preventDefault();
            const url = this.querySelector('input[type="url"]').value;
            if (url) {
                window.location.href = `/tool?demo_url=${encodeURIComponent(url)}`;
            }
        });
    }

    // Initialize any other interactive elements
    const extractForm = document.getElementById('extractForm');
    if (extractForm) {
        extractForm.addEventListener('submit', function(e) {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
                submitBtn.disabled = true;

                // Re-enable button after form submission
                setTimeout(() => {
                    submitBtn.innerHTML = '<i class="fas fa-search me-2"></i>Extract Keywords';
                    submitBtn.disabled = false;
                }, 5000);
            }
        });
    }
});

// Plan upgrade handling
function upgradePlan(planName) {
    if (confirm(`Upgrade to ${planName} plan?`)) {
        window.location.href = `/upgrade/${planName.toLowerCase()}`;
    }
}

// Export functionality (for Pro/Premium users)
function exportResults(format = 'pdf') {
    showToast(`Exporting results as ${format.toUpperCase()}...`, 'info');
    window.location.href = `/export/${format}`;
}

// Copy API example
function copyApiExample() {
    const codeBlock = document.getElementById('apiExample');
    if (codeBlock) {
        navigator.clipboard.writeText(codeBlock.textContent).then(() => {
            // Update button to show success
            const btn = document.getElementById('copyApiBtn');
            if (btn) {
                const originalHTML = btn.innerHTML;
                btn.innerHTML = '<i class="fas fa-check me-1"></i>Copied!';
                btn.classList.remove('btn-outline-secondary');
                btn.classList.add('btn-success');
                
                setTimeout(() => {
                    btn.innerHTML = originalHTML;
                    btn.classList.remove('btn-success');
                    btn.classList.add('btn-outline-secondary');
                }, 2000);
            }
            showToast('API example copied to clipboard!', 'success');
        }).catch(err => {
            console.error('Failed to copy API example: ', err);
            showToast('Failed to copy API example', 'error');
        });
    }
}

function toggleApiKey() {
    const input = document.getElementById('apiKeyInput');
    const icon = document.getElementById('apiKeyIcon');

    if (input && icon) {
        if (input.type === 'password') {
            input.type = 'text';
            icon.className = 'fas fa-eye-slash';
        } else {
            input.type = 'password';
            icon.className = 'fas fa-eye';
        }
    }
}

function copyApiKey() {
    const input = document.getElementById('apiKeyInput');
    if (input && input.value) {
        navigator.clipboard.writeText(input.value).then(() => {
            showToast('API key copied!', 'success');
        }).catch(err => {
            console.error('Failed to copy API key: ', err);
            // Fallback copy method
            const textArea = document.createElement('textarea');
            textArea.value = input.value;
            document.body.appendChild(textArea);
            textArea.select();
            try {
                document.execCommand('copy');
                showToast('API key copied!', 'success');
            } catch (fallbackErr) {
                showToast('Failed to copy API key', 'error');
            }
            document.body.removeChild(textArea);
        });
    } else {
        showToast('No API key to copy', 'warning');
    }
}

function regenerateApiKey() {
    if (confirm('This will invalidate your current API key. Any applications using it will stop working. Continue?')) {
        fetch('/regenerate_api_key', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('apiKeyInput').value = data.api_key;
                // Update the example code
                const codeBlock = document.querySelector('pre code');
                if (codeBlock) {
                    codeBlock.textContent = codeBlock.textContent.replace(/Bearer [^\s"]+/, `Bearer ${data.api_key}`);
                }
                showToast('API key regenerated successfully!');
            } else {
                showToast('Failed to regenerate API key: ' + (data.error || 'Unknown error'), 'error');
            }
        })
        .catch(error => {
            showToast('Error regenerating API key: ' + error.message, 'error');
        });
    }
}

// Toast notification function
function showToast(message, type = 'info') {
    // Create toast element if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }

    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type === 'info' ? 'primary' : type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;

    toastContainer.appendChild(toast);
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();

    // Remove toast after it's hidden
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

// Theme management functions
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    applyTheme(savedTheme);
    
    // Update theme toggle if it exists
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.checked = savedTheme === 'dark';
    }
}

function applyTheme(theme) {
    const htmlRoot = document.documentElement || document.getElementById('htmlRoot');
    const bootstrapTheme = document.getElementById('bootstrapTheme');
    
    if (theme === 'dark') {
        htmlRoot.setAttribute('data-bs-theme', 'dark');
        if (bootstrapTheme) {
            bootstrapTheme.href = 'https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css';
        }
        document.body.style.backgroundColor = '#1a1a1a';
        document.body.style.color = '#ffffff';
    } else {
        htmlRoot.setAttribute('data-bs-theme', 'light');
        if (bootstrapTheme) {
            bootstrapTheme.href = 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css';
        }
        document.body.style.backgroundColor = '#ffffff';
        document.body.style.color = '#000000';
    }
}

function toggleTheme() {
    const currentTheme = localStorage.getItem('theme') || 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    applyTheme(newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Update toggle state
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.checked = newTheme === 'dark';
    }
    
    showToast(`Switched to ${newTheme} mode`, 'success');
}

// Add event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize theme on page load
    initializeTheme();
    
    // Add theme toggle event listener
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('change', function() {
            const newTheme = this.checked ? 'dark' : 'light';
            applyTheme(newTheme);
            localStorage.setItem('theme', newTheme);
            showToast(`Switched to ${newTheme} mode`, 'success');
        });
    }
    
    console.log('SEOExtract app loaded successfully');
});