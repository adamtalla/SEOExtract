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