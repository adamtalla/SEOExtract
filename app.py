
import os
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_cors import CORS
from keyword_extractor import extract_keywords_from_url
from werkzeug.middleware.proxy_fix import ProxyFix
import requests
import json
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Enable CORS for API integration
CORS(app)

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Stripe configuration
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY")

# Plan limits configuration
PLAN_LIMITS = {
    'free': {
        'audits_per_month': 3,
        'keywords_per_audit': 5,
        'seo_suggestions': 0,
        'export': False,
        'api_access': False
    },
    'pro': {
        'audits_per_month': -1,  # Unlimited
        'keywords_per_audit': 10,
        'seo_suggestions': 5,
        'export': True,
        'api_access': False
    },
    'premium': {
        'audits_per_month': -1,  # Unlimited
        'keywords_per_audit': 15,
        'seo_suggestions': 10,
        'export': True,
        'api_access': True
    }
}

# Special access for admin user
ADMIN_EMAIL = "tall3aadam@gmail.com"

def supabase_request(method, endpoint, data=None, auth_token=None):
    """Make a request to Supabase"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    
    headers = {
        'apikey': SUPABASE_KEY,
        'Content-Type': 'application/json'
    }
    
    if auth_token:
        headers['Authorization'] = f'Bearer {auth_token}'
    
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers)
        elif method == 'POST':
            response = requests.post(url, headers=headers, json=data)
        elif method == 'PATCH':
            response = requests.patch(url, headers=headers, json=data)
        else:
            return None
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            logging.error(f"Supabase request failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logging.error(f"Supabase request error: {str(e)}")
        return None

def get_user_from_session():
    """Get user data from session"""
    if 'user_id' in session:
        return {
            'id': session['user_id'],
            'email': session.get('email', ''),
            'plan': session.get('plan', 'free')
        }
    return None

def get_user_usage(user_id):
    """Get user's current month usage"""
    current_month = datetime.now().strftime('%Y-%m')
    
    # Mock usage data for demo - in production, query from Supabase
    return {
        'audits_used': session.get(f'audits_used_{current_month}', 0),
        'month': current_month
    }

def check_usage_limits(user, action='audit'):
    """Check if user can perform action based on their plan limits"""
    if not user:
        return False, "Please log in to continue"
    
    plan = user.get('plan', 'free')
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS['free'])
    usage = get_user_usage(user['id'])
    
    if action == 'audit':
        if limits['audits_per_month'] == -1:  # Unlimited
            return True, ""
        
        if usage['audits_used'] >= limits['audits_per_month']:
            return False, f"You've reached your monthly limit of {limits['audits_per_month']} audits. Please upgrade your plan."
    
    return True, ""

def increment_usage(user_id, action='audit'):
    """Increment user's usage counter"""
    current_month = datetime.now().strftime('%Y-%m')
    
    if action == 'audit':
        key = f'audits_used_{current_month}'
        session[key] = session.get(key, 0) + 1

def login_required(f):
    """Decorator to require login for certain routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    """Landing page with hero section"""
    return render_template('index.html')

@app.route('/tool')
@login_required
def tool():
    """Tool page for keyword extraction"""
    user = get_user_from_session()
    usage = get_user_usage(user['id']) if user else {}
    plan = user.get('plan', 'free') if user else 'free'
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS['free'])
    
    return render_template('tool.html', user=user, usage=usage, limits=limits, user_plan=plan)

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

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page and handler"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # In production, authenticate with Supabase
        # For demo purposes, create a mock login
        if email and password:
            # Mock user data
            session['user_id'] = f"user_{hash(email) % 10000}"
            session['email'] = email
            
            # Special access for admin user
            if email == ADMIN_EMAIL:
                session['plan'] = 'premium'  # Give admin premium access
                flash('Welcome back, Admin! Premium access granted.', 'success')
            else:
                session['plan'] = 'free'  # Default plan
                flash('Login successful!', 'success')
            
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page and handler"""
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # In production, create user in Supabase
        # For demo purposes, create a mock registration
        if name and email and password:
            # Mock user creation
            session['user_id'] = f"user_{hash(email) % 10000}"
            session['email'] = email
            
            # Special access for admin user
            if email == ADMIN_EMAIL:
                session['plan'] = 'premium'  # Give admin premium access
                flash('Welcome! Premium access granted for admin account.', 'success')
            else:
                session['plan'] = 'free'
                flash('Account created successfully!', 'success')
            
            return redirect(url_for('dashboard'))
        else:
            flash('Please fill in all fields', 'danger')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout handler"""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard (requires auth)"""
    user = get_user_from_session()
    usage = get_user_usage(user['id'])
    plan = user.get('plan', 'free')
    
    # Mock recent analyses data
    recent_analyses = []
    
    return render_template('dashboard.html', 
                         user=user, 
                         usage=usage, 
                         user_plan=plan,
                         recent_analyses=recent_analyses)

@app.route('/extract', methods=['POST'])
@login_required
def extract_keywords():
    """Extract keywords from URL - handles both form and JSON requests"""
    try:
        user = get_user_from_session()
        
        # Check usage limits
        can_use, error_msg = check_usage_limits(user, 'audit')
        if not can_use:
            if request.is_json:
                return jsonify({'error': error_msg}), 403
            flash(error_msg, 'warning')
            return redirect(url_for('plans'))
        
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
        
        # Extract keywords
        all_keywords = extract_keywords_from_url(url)
        
        # Apply plan limits
        plan = user.get('plan', 'free')
        limits = PLAN_LIMITS.get(plan, PLAN_LIMITS['free'])
        keywords = all_keywords[:limits['keywords_per_audit']]
        
        # Generate SEO suggestions if plan allows
        seo_suggestions = []
        if limits['seo_suggestions'] > 0:
            # Mock SEO suggestions - in production, use AI to generate these
            suggestions = [
                "Include target keywords in your page title",
                "Add meta description with primary keywords",
                "Use header tags (H1, H2) to structure content",
                "Optimize images with alt text",
                "Improve page loading speed",
                "Add internal links to related content",
                "Create keyword-rich URL slugs",
                "Write compelling meta descriptions",
                "Use schema markup for better visibility",
                "Optimize for mobile responsiveness"
            ]
            seo_suggestions = suggestions[:limits['seo_suggestions']]
        
        # Store results in session for export
        session['last_keywords'] = keywords
        session['last_url'] = url
        session['last_suggestions'] = seo_suggestions
        
        # Increment usage counter
        increment_usage(user['id'], 'audit')
        
        # Return JSON response for API calls
        if request.is_json:
            return jsonify({
                'keywords': keywords, 
                'url': url,
                'seo_suggestions': seo_suggestions,
                'plan': plan
            })
        
        # Return HTML response for form submissions
        return render_template('tool.html', 
                             keywords=keywords, 
                             url=url,
                             seo_suggestions=seo_suggestions,
                             user=user,
                             usage=get_user_usage(user['id']),
                             limits=limits,
                             user_plan=plan)
    
    except Exception as e:
        logging.error(f"Error extracting keywords: {str(e)}")
        error_message = str(e)
        
        if request.is_json:
            return jsonify({'error': error_message}), 500
        
        flash(f"Error analyzing website: {error_message}", 'danger')
        return render_template('tool.html', 
                             error=error_message, 
                             url=url if 'url' in locals() else '',
                             user=user,
                             usage=get_user_usage(user['id']),
                             limits=PLAN_LIMITS.get(user.get('plan', 'free'), PLAN_LIMITS['free']),
                             user_plan=user.get('plan', 'free'))

@app.route('/api/extract_keywords', methods=['POST'])
@login_required
def api_extract_keywords():
    """API endpoint for keyword extraction"""
    try:
        user = get_user_from_session()
        plan = user.get('plan', 'free')
        limits = PLAN_LIMITS.get(plan, PLAN_LIMITS['free'])
        
        # Check API access
        if not limits['api_access']:
            return jsonify({'error': 'API access requires Premium plan'}), 403
        
        # Check usage limits
        can_use, error_msg = check_usage_limits(user, 'audit')
        if not can_use:
            return jsonify({'error': error_msg}), 403
        
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'URL is required in JSON body'}), 400
        
        url = data['url']
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Extract keywords
        all_keywords = extract_keywords_from_url(url)
        keywords = all_keywords[:limits['keywords_per_audit']]
        
        # Generate SEO suggestions
        seo_suggestions = []
        if limits['seo_suggestions'] > 0:
            suggestions = [
                "Include target keywords in your page title",
                "Add meta description with primary keywords",
                "Use header tags (H1, H2) to structure content",
                "Optimize images with alt text",
                "Improve page loading speed"
            ]
            seo_suggestions = suggestions[:limits['seo_suggestions']]
        
        # Increment usage
        increment_usage(user['id'], 'audit')
        
        return jsonify({
            'keywords': keywords, 
            'url': url,
            'seo_suggestions': seo_suggestions,
            'plan': plan,
            'usage': get_user_usage(user['id'])
        })
    
    except Exception as e:
        logging.error(f"API error extracting keywords: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/export/<format>')
@login_required
def export_results(format):
    """Export the last analysis results"""
    user = get_user_from_session()
    plan = user.get('plan', 'free')
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS['free'])
    
    if not limits['export']:
        flash('Export feature requires Pro or Premium plan', 'warning')
        return redirect(url_for('plans'))
    
    # Get last analysis from session
    last_keywords = session.get('last_keywords', [])
    last_url = session.get('last_url', '')
    last_suggestions = session.get('last_suggestions', [])
    
    if not last_keywords:
        flash('No analysis results to export', 'warning')
        return redirect(url_for('tool'))
    
    if format.lower() == 'pdf':
        # Mock PDF export - in production, use reportlab or similar
        flash('PDF export feature coming soon!', 'info')
    elif format.lower() == 'csv':
        from flask import make_response
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Keyword', 'URL', 'Position'])
        for i, keyword in enumerate(last_keywords, 1):
            writer.writerow([keyword, last_url, i])
        
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=seo_keywords_{last_url.replace("https://", "").replace("http://", "").replace("/", "_")[:20]}.csv'
        return response
    
    return redirect(url_for('tool'))

@app.route('/upgrade/<plan>')
@login_required
def upgrade_plan(plan):
    """Upgrade user plan (would integrate with Stripe in production)"""
    user = get_user_from_session()
    
    if plan in ['pro', 'premium']:
        # In production, redirect to Stripe Checkout
        # For demo, just update the session
        session['plan'] = plan
        flash(f'Successfully upgraded to {plan.title()} plan!', 'success')
    else:
        flash('Invalid plan selected', 'danger')
    
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
