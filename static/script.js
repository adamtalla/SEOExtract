// SEOExtract JavaScript functionality

// Copy keywords functionality
function copyKeywords() {
    const keywords = document.querySelectorAll('#keywordsList .badge');
    const keywordTexts = Array.from(keywords).map(badge => 
        badge.textContent.replace(/^\d+\.\s*/, '')
    );

    const textToCopy = keywordTexts.join('\n');

    if (navigator.clipboard) {
        navigator.clipboard.writeText(textToCopy).then(() => {
            showToast('Keywords copied to clipboard!', 'success');
        }).catch(err => {
            console.error('Failed to copy: ', err);
            fallbackCopy(textToCopy);
        });
    } else {
        fallbackCopy(textToCopy);
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

// Show toast notification
function showToast(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `alert alert-${type === 'error' ? 'danger' : 'success'} position-fixed`;
    toast.style.cssText = 'top: 20px; right: 20px; z-index: 1050; min-width: 300px;';
    toast.innerHTML = `
        ${message}
        <button type="button" class="btn-close ms-2" onclick="this.parentElement.remove()"></button>
    `;

    document.body.appendChild(toast);

    // Auto remove after 3 seconds
    setTimeout(() => {
        if (toast.parentElement) {
            toast.remove();
        }
    }, 3000);
}

// Form submission handling
document.addEventListener('DOMContentLoaded', function() {
    const extractForm = document.getElementById('extractForm');
    const extractBtn = document.getElementById('extractBtn');

    if (extractForm && extractBtn) {
        extractForm.addEventListener('submit', function() {
            // Show loading state
            extractBtn.disabled = true;
            extractBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
        });
    }

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
    // In production, this would generate and download the file
    setTimeout(() => {
        showToast(`Export feature coming soon!`, 'info');
    }, 1000);
}