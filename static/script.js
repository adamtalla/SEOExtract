// SEO Keyword Extractor JavaScript functionality

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('extractForm');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const extractBtn = document.getElementById('extractBtn');
    const urlInput = document.getElementById('url');

    // Handle form submission
    form.addEventListener('submit', function(e) {
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        extractBtn.disabled = true;
        extractBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
        
        // Scroll to loading indicator
        loadingIndicator.scrollIntoView({ behavior: 'smooth' });
    });

    // Auto-format URL input
    urlInput.addEventListener('blur', function() {
        let url = this.value.trim();
        if (url && !url.startsWith('http://') && !url.startsWith('https://')) {
            this.value = 'https://' + url;
        }
    });

    // Handle Enter key in URL input
    urlInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            form.submit();
        }
    });
});

// Copy keywords to clipboard
function copyKeywords() {
    const keywords = [];
    const keywordElements = document.querySelectorAll('.keyword-text');
    
    keywordElements.forEach(function(element) {
        keywords.push(element.textContent);
    });
    
    const keywordText = keywords.join(', ');
    
    // Use the modern clipboard API if available
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(keywordText).then(function() {
            showCopySuccess();
        }).catch(function() {
            fallbackCopy(keywordText);
        });
    } else {
        fallbackCopy(keywordText);
    }
}

// Fallback copy method for older browsers
function fallbackCopy(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.opacity = '0';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        document.execCommand('copy');
        showCopySuccess();
    } catch (err) {
        console.error('Copy failed:', err);
        showCopyError();
    }
    
    document.body.removeChild(textArea);
}

// Show copy success message
function showCopySuccess() {
    const button = event.target;
    const originalText = button.innerHTML;
    
    button.innerHTML = '<i class="fas fa-check me-2"></i>Copied!';
    button.classList.remove('btn-outline-secondary');
    button.classList.add('btn-success');
    
    setTimeout(function() {
        button.innerHTML = originalText;
        button.classList.remove('btn-success');
        button.classList.add('btn-outline-secondary');
    }, 2000);
}

// Show copy error message
function showCopyError() {
    const button = event.target;
    const originalText = button.innerHTML;
    
    button.innerHTML = '<i class="fas fa-times me-2"></i>Copy Failed';
    button.classList.remove('btn-outline-secondary');
    button.classList.add('btn-danger');
    
    setTimeout(function() {
        button.innerHTML = originalText;
        button.classList.remove('btn-danger');
        button.classList.add('btn-outline-secondary');
    }, 2000);
}

// Auto-dismiss alerts after 5 seconds
document.addEventListener('DOMContentLoaded', function() {
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
});
