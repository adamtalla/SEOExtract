
import os
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
from keyword_extractor import extract_keywords_from_url
from werkzeug.middleware.proxy_fix import ProxyFix

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Enable CORS for API integration
CORS(app)

@app.route('/')
def index():
    """Landing page with hero section"""
    return render_template('index.html')

@app.route('/tool')
def tool():
    """Tool page for keyword extraction"""
    return render_template('tool.html')

@app.route('/plans')
def plans():
    """Pricing plans page"""
    return render_template('plans.html')

@app.route('/faq')
def faq():
    """FAQ page"""
    return render_template('faq.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/login')
def login():
    """Login page"""
    return render_template('login.html')

@app.route('/register')
def register():
    """Registration page"""
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    """User dashboard (requires auth)"""
    # TODO: Add authentication check
    return render_template('dashboard.html')

@app.route('/extract', methods=['POST'])
def extract_keywords():
    """Extract keywords from URL - handles both form and JSON requests"""
    try:
        # Handle both form data and JSON data
        if request.is_json:
            data = request.get_json()
            url = data.get('url')
        else:
            url = request.form.get('url')
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # TODO: Add authentication and usage limit checks here
        
        # Extract keywords
        keywords = extract_keywords_from_url(url)
        
        # Return JSON response for API calls
        if request.is_json:
            return jsonify({'keywords': keywords, 'url': url})
        
        # Return HTML response for form submissions
        return render_template('tool.html', keywords=keywords, url=url)
    
    except Exception as e:
        logging.error(f"Error extracting keywords: {str(e)}")
        error_message = str(e)
        
        if request.is_json:
            return jsonify({'error': error_message}), 500
        
        return render_template('tool.html', error=error_message, url=url if 'url' in locals() else '')

@app.route('/api/extract_keywords', methods=['POST'])
def api_extract_keywords():
    """API endpoint for keyword extraction"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'URL is required in JSON body'}), 400
        
        url = data['url']
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # TODO: Add authentication and usage limit checks here
        
        # Extract keywords
        keywords = extract_keywords_from_url(url)
        
        return jsonify({'keywords': keywords, 'url': url})
    
    except Exception as e:
        logging.error(f"API error extracting keywords: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
