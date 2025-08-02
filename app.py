# --- AI-Powered Keyword Extraction ---
# The keyword quality filtering is now handled by the AI keyword extractor
# which uses semantic analysis and NLP techniques for better results
import os
# Load environment variables from .env at the very top
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[WARNING] python-dotenv not installed. .env will not be loaded automatically.")

import logging
import secrets
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
app.secret_key = os.environ.get("SECRET_KEY",
                                "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Enable CORS for API integration
CORS(app)

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
# Debug print to verify env loading (do not log secrets in production)
print("[DEBUG] SUPABASE_URL:", SUPABASE_URL)
print("[DEBUG] SUPABASE_KEY:", "SET" if SUPABASE_KEY else "NOT SET")

# Stripe configuration
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY")

# Plan limits configuration
PLAN_LIMITS = {
    'free': {
        'audits_per_month': 3,  # 3 audits per month for free plan
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

from keyword_extractor import extract_keywords_from_url, get_detailed_keywords
from seo_analyzer import generate_seo_suggestions
from seo_audit import SEOAuditor

# Import training system
try:
    from seo_keyword_trainer import get_keyword_trainer
    TRAINER_AVAILABLE = True
except ImportError:
    TRAINER_AVAILABLE = False
    logging.warning("SEO keyword trainer not available")


def supabase_request(method, endpoint, data=None, auth_token=None):
    """Make a request to Supabase"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None

    headers = {'apikey': SUPABASE_KEY, 'Content-Type': 'application/json'}

    if auth_token:
        headers['Authorization'] = f'Bearer {auth_token}'

    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"

    try:
        logging.debug(f"Supabase {method} {url} data={data}")
        if method == 'GET':
            response = requests.get(url, headers=headers)
        elif method == 'POST':
            response = requests.post(url, headers=headers, json=data)
        elif method == 'PATCH':
            response = requests.patch(url, headers=headers, json=data)
        else:
            logging.error(f"Supabase request: Unsupported method {method}")
            return {'error': f'Unsupported method {method}'}

        logging.debug(f"Supabase response: {response.status_code} {response.text}")
        if response.status_code in [200, 201]:
            # If response body is empty, treat as success (common for POST)
            if not response.text or response.text.strip() == '':
                return '__SUPABASE_EMPTY_SUCCESS__'
            try:
                return response.json()
            except Exception as e:
                logging.error(f"Error parsing Supabase JSON: {str(e)}")
                # If response is empty, treat as success
                if not response.text or response.text.strip() == '':
                    return '__SUPABASE_EMPTY_SUCCESS__'
                return {'error': f'Error parsing JSON: {str(e)}', 'raw': response.text}
        else:
            logging.error(f"Supabase request failed: {response.status_code} - {response.text}")
            return {'error': f'Supabase request failed', 'status_code': response.status_code, 'response': response.text}
    except Exception as e:
        logging.error(f"Supabase request error: {str(e)}")
        return {'error': f'Exception: {str(e)}'}


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
    """Get user's current month usage from Supabase"""
    current_month = datetime.now().strftime('%Y-%m')

    if not user_id or user_id == 'guest':
        return {'audits_used': 0, 'month': current_month}

    # Try to get usage from Supabase
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            usage_data = supabase_request('GET', f'user_usage?user_id=eq.{user_id}&month=eq.{current_month}')
            if usage_data and len(usage_data) > 0:
                return {
                    'audits_used': usage_data[0].get('audits_used', 0),
                    'month': current_month
                }
            else:
                # Create initial usage record for this month
                initial_usage = {
                    'user_id': user_id,
                    'month': current_month,
                    'audits_used': 0,
                    'keywords_generated': 0,
                    'exports_used': 0,
                    'api_calls_used': 0
                }
                supabase_request('POST', 'user_usage', initial_usage)
                return {'audits_used': 0, 'month': current_month}
        except Exception as e:
            logging.error(f"Error getting user usage from Supabase: {str(e)}")

    # Fallback to session-based tracking (for backwards compatibility)
    return {
        'audits_used': session.get(f'audits_used_{user_id}_{current_month}', 0),
        'month': current_month
    }


def check_usage_limits(user, action='audit'):
    """Check if user can perform action based on their plan limits"""
    if not user:
        return False, "Please log in to continue"

    user_id = user.get('id', '')
    plan = user.get('plan', 'free')
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS['free'])
    usage = get_user_usage(user_id)

    if action == 'audit':
        if limits['audits_per_month'] == -1:  # Unlimited
            logging.info(f"User {user_id} has unlimited audits ({plan} plan)")
            return True, ""

        audits_used = usage.get('audits_used', 0)
        audits_limit = limits['audits_per_month']
        remaining = audits_limit - audits_used

        logging.info(f"User {user_id} audit check: {audits_used}/{audits_limit} used, {remaining} remaining")

        if audits_used >= audits_limit:
            logging.warning(f"User {user_id} has reached audit limit: {audits_used}/{audits_limit}")
            return False, f"You've reached your monthly limit of {audits_limit} audits. Please upgrade your plan."

        # Warning when getting close to limit
        if remaining <= 1 and remaining > 0:
            logging.info(f"User {user_id} is close to audit limit: {remaining} remaining")

    return True, ""


def store_search_history(user_id, url, keywords, seo_suggestions, seo_data):
    """Store search history in database"""
    if not user_id or user_id == 'guest':
        return

    # Try to store in Supabase
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            history_data = {
                'user_id': user_id,
                'url': url,
                'keywords': keywords,
                'seo_suggestions': seo_suggestions,
                'seo_data': seo_data,
                'keyword_count': len(keywords) if isinstance(keywords, list) else 0,
                'created_at': datetime.now().isoformat()
            }

            result = supabase_request('POST', 'search_history', history_data)
            if result:
                logging.info(f"Stored search history for user {user_id}, URL: {url}")
            else:
                logging.error(f"Failed to store search history for user {user_id}")

        except Exception as e:
            logging.error(f"Error storing search history: {str(e)}")
            # Fallback to session storage
            if 'search_history' not in session:
                session['search_history'] = []

            session['search_history'].append({
                'url': url,
                'keywords': keywords,
                'seo_suggestions': seo_suggestions,
                'seo_data': seo_data,
                'keyword_count': len(keywords) if isinstance(keywords, list) else 0,
                'created_at': datetime.now().isoformat()
            })

            # Keep only last 50 searches in session
            if len(session['search_history']) > 50:
                session['search_history'] = session['search_history'][-50:]
    else:
        # Session-based storage fallback
        if 'search_history' not in session:
            session['search_history'] = []

        session['search_history'].append({
            'url': url,
            'keywords': keywords,
            'seo_suggestions': seo_suggestions,
            'seo_data': seo_data,
            'keyword_count': len(keywords) if isinstance(keywords, list) else 0,
            'created_at': datetime.now().isoformat()
        })

        # Keep only last 50 searches in session
        if len(session['search_history']) > 50:
            session['search_history'] = session['search_history'][-50:]


def get_search_history(user_id):
    """Get user's search history"""
    if not user_id or user_id == 'guest':
        return []

    # Try to get from Supabase
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            history = supabase_request('GET', f'search_history?user_id=eq.{user_id}&order=created_at.desc&limit=50')
            if history:
                return history
        except Exception as e:
            logging.error(f"Error getting search history from Supabase: {str(e)}")

    # Fallback to session
    return session.get('search_history', [])


def increment_usage(user_id, action='audit'):
    """Increment user's usage counter in database"""
    if not user_id or user_id == 'guest':
        return

    current_month = datetime.now().strftime('%Y-%m')

    # Try to update usage in Supabase
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            # Get current usage
            current_usage = get_user_usage(user_id)
            new_count = current_usage['audits_used'] + 1

            # Update the usage record
            update_data = {}
            if action == 'audit':
                update_data['audits_used'] = new_count
            elif action == 'export':
                update_data['exports_used'] = current_usage.get('exports_used', 0) + 1
            elif action == 'api':
                update_data['api_calls_used'] = current_usage.get('api_calls_used', 0) + 1

            # Update existing record or create new one
            existing_usage = supabase_request('GET', f'user_usage?user_id=eq.{user_id}&month=eq.{current_month}')

            if existing_usage and len(existing_usage) > 0:
                # Update existing record
                supabase_request('PATCH', f'user_usage?user_id=eq.{user_id}&month=eq.{current_month}', update_data)
            else:
                # Create new record
                new_usage = {
                    'user_id': user_id,
                    'month': current_month,
                    'audits_used': 1 if action == 'audit' else 0,
                    'keywords_generated': 0,
                    'exports_used': 1 if action == 'export' else 0,
                    'api_calls_used': 1 if action == 'api' else 0
                }
                supabase_request('POST', 'user_usage', new_usage)

            logging.info(f"Updated usage for user {user_id}: {action} count incremented")

        except Exception as e:
            logging.error(f"Error updating usage in Supabase: {str(e)}")
            # Fallback to session-based tracking
            if action == 'audit':
                key = f'audits_used_{user_id}_{current_month}'
                session[key] = session.get(key, 0) + 1
    else:
        # Fallback to session-based tracking
        if action == 'audit':
            key = f'audits_used_{user_id}_{current_month}'
            session[key] = session.get(key, 0) + 1


def generate_api_key():
    """Generate a secure API key"""
    return f"sk-{secrets.token_urlsafe(32)}"


def get_user_by_api_key(api_key):
    """Get user data from API key"""
    if not api_key:
        return None

    # First try Supabase if available
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            # Query user_profiles table for the API key
            user_data = supabase_request('GET', f'user_profiles?api_key=eq.{api_key}&select=id,email,plan,subscription_status')
            if user_data and len(user_data) > 0:
                user = user_data[0]
                return {
                    'id': user['id'],
                    'email': user['email'],
                    'plan': user['plan']
                }
        except Exception as e:
            logging.error(f"Error getting user by API key from Supabase: {str(e)}")

    # Fallback: Check session-based API keys
    # This is a simplified approach - in production you'd want a proper database
    # For demo purposes, we'll use a basic admin check
    if api_key.startswith('sk-') and len(api_key) > 20:
        # Return admin user data for valid-looking API keys in demo mode
        return {
            'id': f"user_{hash(ADMIN_EMAIL) % 10000}",
            'email': ADMIN_EMAIL,
            'plan': 'premium'
        }

    return None


def api_key_required(f):
    """Decorator to require API key for API endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')

        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid Authorization header. Use: Authorization: Bearer YOUR_API_KEY'}), 401

        api_key = auth_header.replace('Bearer ', '').strip()
        user = get_user_by_api_key(api_key)

        if not user:
            return jsonify({'error': 'Invalid API key'}), 401

        # Store user in request context
        request.current_user = user
        return f(*args, **kwargs)

    return decorated_function


def initialize_new_user(user_id, email, plan='free'):
    """Initialize a new user with their starting audit allowance"""
    if not user_id or not email:
        return False
    # This function is now only called from registration, so password is required
    return True  # No-op, registration handles Supabase now


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

    return render_template('tool.html',
                           user=user,
                           usage=usage,
                           limits=limits,
                           user_plan=plan)


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
    """Login page and handler (Supabase authentication)"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if email and password:
            # Query Supabase for user with this email
            user_data = None
            if SUPABASE_URL and SUPABASE_KEY:
                try:
                    users = supabase_request('GET', f'user_profiles?email=eq.{email}')
                    if users and len(users) > 0:
                        user_data = users[0]
                except Exception as e:
                    logging.error(f"Error querying Supabase for login: {str(e)}")

            if user_data:
                # Check password (hashed)
                import hashlib
                hashed_pw = hashlib.sha256(password.encode()).hexdigest()
                if str(user_data.get('password', '')) == hashed_pw:
                    user_id = user_data.get('id')
                    session['user_id'] = user_id
                    session['email'] = email
                    session['plan'] = user_data.get('plan', 'free')
                    flash('Login successful!', 'success')
                    return redirect(url_for('dashboard'))
                else:
                    flash('Incorrect password. Please try again.', 'danger')
            else:
                flash('No account found with that email.', 'danger')
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

        if not (name and email and password):
            flash('Please fill in all fields', 'danger')
            return render_template('register.html')

        # Always create user in Supabase
        import hashlib
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        user_id = f"user_{hash(email) % 10000}"
        plan = 'premium' if email == ADMIN_EMAIL else 'free'
        user_profile = {
            'id': user_id,
            'email': email,
            'password': hashed_pw,
            'plan': plan,
            'subscription_status': 'active' if plan == 'premium' else 'inactive',
            'auto_upgrade': True,
            'api_key': generate_api_key(),
            'created_at': datetime.now().isoformat()
        }

        # Check if user already exists
        user_exists = False
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                users = supabase_request('GET', f'user_profiles?email=eq.{email}')
                if users and len(users) > 0:
                    user_exists = True
            except Exception as e:
                logging.error(f"Error checking user in Supabase: {str(e)}")

        if user_exists:
            flash('An account with that email already exists. Please log in.', 'warning')
            return redirect(url_for('login'))



        # Create user in Supabase
        created = False
        supabase_error = None
        result = None  # Ensure result is always defined
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                result = supabase_request('POST', 'user_profiles', user_profile)
                logging.debug(f"[REGISTER] Supabase user creation response: {result} (type: {type(result)})")
                # Treat as success if:
                # - result is the special empty success marker
                # - result is None (Supabase returns no content)
                # - result is a dict and has no 'error'
                # - result is an empty list (Supabase returns [] on success)
                # - result is an empty dict
                if result == '__SUPABASE_EMPTY_SUCCESS__':
                    created = True
                elif result is None:
                    created = True
                elif (isinstance(result, dict) and not result.get('error')):
                    created = True
                elif (isinstance(result, list) and len(result) == 0):
                    created = True
                elif (isinstance(result, dict) and len(result) == 0):
                    created = True
                else:
                    supabase_error = result.get('error') if isinstance(result, dict) else str(result)
                    # Also include status code and response if present
                    if isinstance(result, dict):
                        if 'status_code' in result:
                            supabase_error += f" (status: {result['status_code']})"
                        if 'response' in result:
                            supabase_error += f" | response: {result['response']}"
            except Exception as e:
                logging.error(f"Error creating user in Supabase: {str(e)}")
                supabase_error = str(e)

        if created:
            session['user_id'] = user_id
            session['email'] = email
            session['plan'] = plan
            # Also create initial usage record
            try:
                current_month = datetime.now().strftime('%Y-%m')
                initial_usage = {
                    'user_id': user_id,
                    'month': current_month,
                    'audits_used': 0,
                    'keywords_generated': 0,
                    'exports_used': 0,
                    'api_calls_used': 0
                }
                supabase_request('POST', 'user_usage', initial_usage)
            except Exception as e:
                logging.error(f"Error creating initial usage in Supabase: {str(e)}")
            flash('Account created successfully! You have 5 free audits to get started.', 'success')
            return redirect(url_for('dashboard'))
        else:
            logging.error(f"Supabase user creation failed. Supabase error: {supabase_error}. Data: {user_profile}")
            logging.error(f"Full Supabase response: {result}")
            flash(f'Error creating account. Please try again. (Supabase error: {supabase_error})', 'danger')
            return render_template('register.html')

    return render_template('register.html')


@app.route('/logout')
def logout():
    """Logout handler"""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/settings')
@login_required
def settings():
    """User settings page"""
    user = get_user_from_session()
    plan = user.get('plan', 'free')
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS['free'])

    return render_template('settings.html',
                           user=user,
                           user_plan=plan,
                           limits=limits)


@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard (requires auth)"""
    user = get_user_from_session()
    usage = get_user_usage(user['id'])
    plan = user.get('plan', 'free')
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS['free'])

    # Get user's API key if they have API access (Premium only)
    api_key = None
    if limits['api_access'] and plan == 'premium':
        try:
            # First try to get from session (fallback)
            api_key = session.get(f'api_key_{user["id"]}')

            # Then try Supabase if available
            if SUPABASE_URL and SUPABASE_KEY:
                try:
                    user_data = supabase_request('GET', f'user_profiles?id=eq.{user["id"]}&select=api_key')
                    if user_data and len(user_data) > 0:
                        supabase_api_key = user_data[0].get('api_key')
                        if supabase_api_key:
                            api_key = supabase_api_key
                            # Update session cache
                            session[f'api_key_{user["id"]}'] = api_key
                except Exception as e:
                    logging.error(f"Error getting API key from Supabase: {str(e)}")

            # Generate API key if user doesn't have one
            if not api_key:
                api_key = generate_api_key()
                session[f'api_key_{user["id"]}'] = api_key

                # Try to store in Supabase if available
                if SUPABASE_URL and SUPABASE_KEY:
                    try:
                        update_data = {
                            'api_key': api_key,
                            'updated_at': datetime.now().isoformat()
                        }
                        supabase_request('PATCH', f'user_profiles?id=eq.{user["id"]}', update_data)
                        logging.info(f"Generated and stored new API key for Premium user {user['email']}")
                    except Exception as e:
                        logging.error(f"Error storing API key in Supabase: {str(e)}")
                        logging.info("API key stored in session as fallback")
                else:
                    logging.info(f"Generated new API key for Premium user {user['email']} (session-based)")

        except Exception as e:
            logging.error(f"Error getting/generating API key: {str(e)}")
            # Generate a basic API key as final fallback
            api_key = generate_api_key()
            session[f'api_key_{user["id"]}'] = api_key

    # Get recent analyses from history
    recent_analyses = get_search_history(user['id'])

    return render_template('dashboard.html',
                           user=user,
                           usage=usage,
                           user_plan=plan,
                           limits=limits,
                           api_key=api_key,
                           recent_analyses=recent_analyses)


@app.route('/extract', methods=['POST'])
@login_required
def extract_keywords():
    """Extract keywords from URL - handles both form and JSON requests"""

    # STEP 1: Initialize EVERYTHING at the very top
    url = ''
    keywords = []
    seo_suggestions = []
    audit_results = []
    seo_data = {}
    user = None
    plan = 'free'
    limits = PLAN_LIMITS['free']
    error_message = ''

    # STEP 2: Set up basic user info
    try:
        user = get_user_from_session()
        if user:
            plan = user.get('plan', 'free')
            limits = PLAN_LIMITS.get(plan, PLAN_LIMITS['free'])
    except Exception as e:
        logging.error(f"Error getting user session: {str(e)}")
        user = {'id': 'guest', 'plan': 'free'}
        plan = 'free'
        limits = PLAN_LIMITS['free']

    # STEP 3: Get URL from request
    try:
        if request.is_json:
            data = request.get_json() or {}
            url = data.get('url', '')
        else:
            url = request.form.get('url', '')
    except Exception as e:
        logging.error(f"Error getting URL from request: {str(e)}")
        url = ''

    # STEP 4: Basic validation
    if not url:
        error_message = 'URL is required'
        if request.is_json:
            return jsonify({'error': error_message}), 400
        flash(error_message, 'danger')
        return render_template('tool.html',
                               error=error_message,
                               url=url,
                               keywords=keywords,
                               seo_suggestions=seo_suggestions,
                               audit_results=audit_results,
                               seo_data=seo_data,
                               user=user,
                               usage={
                                   'audits_used': 0,
                                   'month': ''
                               },
                               limits=limits,
                               user_plan=plan)

    # STEP 5: Check usage limits
    try:
        can_use, limit_error = check_usage_limits(user, 'audit')
        if not can_use:
            if request.is_json:
                return jsonify({'error': limit_error}), 403
            flash(limit_error, 'warning')
            return redirect(url_for('plans'))
    except Exception as e:
        logging.error(f"Error checking usage limits: {str(e)}")
        # Continue anyway for basic functionality

    # STEP 6: Fix URL format
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
    except Exception as e:
        logging.error(f"Error fixing URL format: {str(e)}")

    # STEP 7: Try to extract keywords (SAFELY)
    try:
        # Import here to avoid import errors affecting the whole route
        from keyword_extractor import extract_keywords_from_url


        # Try keyword extraction
        extracted_keywords = extract_keywords_from_url(
            url, limits['keywords_per_audit'])

        # Filter for high-quality keywords
        if isinstance(extracted_keywords, list):
            # Optionally, pass known_products/headings here if available
            keywords = [kw for kw in extracted_keywords]
        else:
            keywords = []
            logging.warning(
                f"extract_keywords_from_url returned non-list: {type(extracted_keywords)}"
            )

    except ImportError as e:
        logging.error(f"Import error for keyword_extractor: {str(e)}")
        keywords = []
        error_message = "Keyword extraction service unavailable"
    except Exception as e:
        logging.error(f"Error extracting keywords: {str(e)}")
        keywords = []
        error_message = f"Could not analyze website: {str(e)}"

    # STEP 8: Try to get SEO data (SAFELY)
    try:
        from web_scraper import get_seo_metadata
        seo_data = get_seo_metadata(url) or {}
    except ImportError as e:
        logging.error(f"Import error for web_scraper: {str(e)}")
        seo_data = {}
    except Exception as e:
        logging.error(f"Error getting SEO metadata: {str(e)}")
        seo_data = {}

    # STEP 9: Try to generate SEO suggestions (SAFELY)
    try:
        if limits['seo_suggestions'] > 0:
            # Use AI-powered suggestions for premium users, rule-based for others
            from ai_seo_analyzer import AISEOAnalyzer

            ai_analyzer = AISEOAnalyzer()
            # Ensure keywords is always a list
            keywords_list = keywords if isinstance(keywords, list) else []
            seo_suggestions = ai_analyzer.generate_hybrid_suggestions(
                url, seo_data, keywords_list, plan)

            # Ensure seo_suggestions is a list and limit
            if not isinstance(seo_suggestions, list):
                seo_suggestions = []
            seo_suggestions = seo_suggestions[:limits['seo_suggestions']]

            # Also generate detailed audit results for export
            try:
                from seo_audit import SEOAuditor
                auditor = SEOAuditor()
                    # Ensure keywords_list is properly defined
                    keywords_list = keywords if isinstance(keywords, list) else []
                    content_text = seo_data.get('content_text', '') if seo_data else ''

                    audit_results = auditor.analyze_page(
                        seo_data, keywords_list, content_text)

                    # Ensure audit_results is a list and limit
                    if not isinstance(audit_results, list):
                        audit_results = []
                    audit_results = audit_results[:limits['seo_suggestions']]
                except Exception as audit_error:
                    logging.error(f"Error generating audit results: {str(audit_error)}")
                    audit_results = []

        else:
            # Free plan - no suggestions
            seo_suggestions = []
            audit_results = []

    except ImportError as e:
        logging.error(f"Import error for AI analyzer: {str(e)}")
        # Fallback to existing system
        try:
            if limits['seo_suggestions'] > 0:
                from seo_audit import SEOAuditor
                auditor = SEOAuditor()
                keywords_list = keywords if isinstance(keywords, list) else []
                audit_results = auditor.analyze_page(
                    seo_data, keywords_list, seo_data.get('content_text', ''))

                if isinstance(audit_results, list):
                    audit_results = audit_results[:limits['seo_suggestions']]
                    for result in audit_results:
                        if isinstance(result, dict):
                            seo_suggestions.append({
                                'type': result.get('type', 'General'),
                                'priority': result.get('priority', 'Medium'),
                                'suggestion': f"{result.get('issue', 'Issue')}: {result.get('recommendation', 'No recommendation')}"
                            })
                else:
                    audit_results = []
        except Exception as fallback_error:
            logging.error(f"Fallback error: {str(fallback_error)}")
            audit_results = []
            seo_suggestions = []
    except Exception as e:
        logging.error(f"Error generating SEO suggestions: {str(e)}")
        audit_results = []
        seo_suggestions = []

    # STEP 10: Store results in session and history (SAFELY)
    try:
        session['last_keywords'] = keywords
        session['last_url'] = url
        session['last_suggestions'] = seo_suggestions
        session['last_audit_results'] = audit_results
        session['last_seo_data'] = seo_data

        # Store in search history
        store_search_history(user['id'], url, keywords, seo_suggestions, seo_data)

        # Increment usage
        increment_usage(user['id'], 'audit')
    except Exception as e:
        logging.error(f"Error storing session data: {str(e)}")

    # STEP 11: Prepare response data
    try:
        usage = get_user_usage(user['id']) if user and user.get('id') else {
            'audits_used': 0,
            'month': ''
        }
    except Exception as e:
        logging.error(f"Error getting usage: {str(e)}")
        usage = {'audits_used': 0, 'month': ''}

    # STEP 12: Return response
    try:
        if request.is_json:
            return jsonify({
                'keywords':
                keywords,
                'url':
                url,
                'seo_suggestions':
                seo_suggestions,
                'audit_results':
                audit_results,
                'seo_data':
                seo_data,
                'plan':
                plan,
                'success':
                len(keywords) > 0 or len(audit_results) > 0
            })

        # HTML response
        return render_template('tool.html',
                               keywords=keywords,
                               url=url,
                               seo_suggestions=seo_suggestions,
                               audit_results=audit_results,
                               seo_data=seo_data,
                               user=user,
                               usage=usage,
                               limits=limits,
                               user_plan=plan,
                               error=error_message if error_message else None)

    except Exception as e:
        logging.error(f"Error rendering response: {str(e)}")
        # Final fallback
        if request.is_json:
            return jsonify({
                'error': 'Internal server error',
                'keywords': [],
                'url': url
            }), 500

        flash('An error occurred while processing your request.', 'danger')
        return render_template(
            'tool.html',
            error='An error occurred while processing your request.',
            url=url,
            keywords=[],
            seo_suggestions=[],
            audit_results=[],
            seo_data={},
            user=user or {'plan': 'free'},
            usage={
                'audits_used': 0,
                'month': ''
            },
            limits=PLAN_LIMITS['free'],
            user_plan='free')


@app.route('/api/extract_keywords', methods=['POST'])
@api_key_required
def api_extract_keywords():
    """API endpoint for keyword extraction using API key authentication"""
    try:
        user = request.current_user
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

        # Extract keywords and SEO metadata
        from web_scraper import get_seo_metadata

        all_keywords = extract_keywords_from_url(url, limits['keywords_per_audit'])
        seo_data = get_seo_metadata(url)
        keywords = all_keywords[:limits['keywords_per_audit']] if isinstance(all_keywords, list) else []

        # Generate SEO suggestions using AI analyzer
        seo_suggestions = []
        audit_results = []

        if limits['seo_suggestions'] > 0:
            try:
                from ai_seo_analyzer import AISEOAnalyzer
                ai_analyzer = AISEOAnalyzer()
                seo_suggestions = ai_analyzer.generate_hybrid_suggestions(url, seo_data, keywords, plan)

                # Limit suggestions based on plan
                if isinstance(seo_suggestions, list):
                    seo_suggestions = seo_suggestions[:limits['seo_suggestions']]
                else:
                    seo_suggestions = []

                # Also generate audit results
                from seo_audit import SEOAuditor
                auditor = SEOAuditor()
                audit_results = auditor.analyze_page(seo_data, keywords, seo_data.get('content_text', ''))
                if isinstance(audit_results, list):
                    audit_results = audit_results[:limits['seo_suggestions']]
                else:
                    audit_results = []

            except Exception as seo_error:
                logging.error(f"Error generating SEO suggestions: {str(seo_error)}")
                seo_suggestions = []
                audit_results = []

        # Increment usage
        increment_usage(user['id'], 'audit')

        return jsonify({
            'keywords': keywords,
            'url': url,
            'seo_suggestions': seo_suggestions,
            'audit_results': audit_results,
            'plan': plan,
            'usage': get_user_usage(user['id']),
            'success': True
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
    last_audit_results = session.get('last_audit_results', [])
    last_seo_data = session.get('last_seo_data', {})

    if not last_keywords and not last_audit_results:
        flash('No analysis results to export', 'warning')
        return redirect(url_for('tool'))

    if format.lower() == 'pdf':
        # Generate PDF export
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            from flask import make_response
            import io

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

            # Container for the 'Flowable' objects
            elements = []

            # Define styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=30)
            heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=14, spaceAfter=12)

            # Add title
            elements.append(Paragraph(f"SEO Analysis Report", title_style))
            elements.append(Paragraph(f"URL: {last_url}", styles['Normal']))
            elements.append(Spacer(1, 12))

            # Add keywords section
            if last_keywords:
                elements.append(Paragraph("Keywords Found:", heading_style))
                keyword_data = []
                for i, keyword in enumerate(last_keywords, 1):
                    keyword_data.append([str(i), keyword])

                keyword_table = Table(keyword_data, colWidths=[0.5*inch, 4*inch])
                keyword_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                    ('ALIGN',(0,0),(-1,-1),'LEFT'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,0), 12),
                    ('BOTTOMPADDING', (0,0), (-1,0), 12),
                    ('BACKGROUND',(0,1),(-1,-1),colors.beige),
                    ('GRID',(0,0),(-1,-1),1,colors.black)
                ]))
                elements.append(keyword_table)
                elements.append(Spacer(1, 12))

            # Add SEO suggestions section
            if last_suggestions:
                elements.append(Paragraph("SEO Suggestions:", heading_style))
                for suggestion in last_suggestions:
                    elements.append(Paragraph(f"<b>Priority:</b> {suggestion.get('priority', 'Medium')}", styles['Normal']))
                    elements.append(Paragraph(f"<b>Type:</b> {suggestion.get('type', 'General')}", styles['Normal']))
                    elements.append(Paragraph(f"<b>Suggestion:</b> {suggestion.get('suggestion', 'No suggestion')}", styles['Normal']))
                    elements.append(Spacer(1, 12))

            # Build PDF
            doc.build(elements)
            pdf_data = buffer.getvalue()
            buffer.close()

            response = make_response(pdf_data)
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = f'attachment; filename=seo_report_{last_url.replace("https://", "").replace("http://", "").replace("/", "_")[:20]}.pdf'
            return response

        except ImportError:
            # Fallback if reportlab is not available
            flash('PDF generation requires additional packages. Please contact support.', 'warning')
        except Exception as e:
            logging.error(f"Error generating PDF: {str(e)}")
            flash('Error generating PDF report', 'danger')
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        if name and email and password:
            # Check if user already exists
            user_exists = False
            user_id = None
            if SUPABASE_URL and SUPABASE_KEY:
                try:
                    users = supabase_request('GET', f'user_profiles?email=eq.{email}')
                    if users and len(users) > 0:
                        user_exists = True
                        user_id = users[0].get('id')
                except Exception as e:
                    logging.error(f"Error checking user in Supabase: {str(e)}")

            if user_exists:
                flash('An account with that email already exists. Please log in.', 'warning')
                return redirect(url_for('login'))

            # Create user in Supabase
            import hashlib
            hashed_pw = hashlib.sha256(password.encode()).hexdigest()
            user_id = f"user_{hash(email) % 10000}"
            plan = 'premium' if email == ADMIN_EMAIL else 'free'
            user_profile = {
                'id': user_id,
                'email': email,
                'password': hashed_pw,
                'plan': plan,
                'subscription_status': 'active' if plan == 'premium' else 'inactive',
                'created_at': datetime.now().isoformat()
            }
            created = False
            if SUPABASE_URL and SUPABASE_KEY:
                try:
                    result = supabase_request('POST', 'user_profiles', user_profile)
                    if result:
                        created = True
                except Exception as e:
                    logging.error(f"Error creating user in Supabase: {str(e)}")

            if created:
                session['user_id'] = user_id
                session['email'] = email
                session['plan'] = plan
                flash('Account created successfully! You have 5 free audits to get started.', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Error creating account. Please try again.', 'danger')
        else:
            flash('Please fill in all fields', 'danger')

    return render_template('register.html')


@app.route('/upgrade/<plan>')
@login_required
def upgrade_plan(plan):
    """Upgrade user plan (would integrate with Stripe in production)"""
    user = get_user_from_session()

    if plan in ['pro', 'premium']:
        # In production, redirect to Stripe Checkout
        # For demo, just update the session and call Supabase function
        session['plan'] = plan

        # Call Supabase function to upgrade plan
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                # Call the upgrade function
                upgrade_data = {
                    'user_email': user.get('email'),
                    'new_plan': plan,
                    'period_end': (datetime.now() + timedelta(days=30)).isoformat()
                }

                response = supabase_request('POST', 'rpc/upgrade_user_plan', upgrade_data)
                if response:
                    logging.info(f"Plan upgraded in Supabase for {user.get('email')} to {plan}")
                else:
                    logging.error(f"Failed to upgrade plan in Supabase for {user.get('email')}")

            except Exception as e:
                logging.error(f"Error upgrading plan in Supabase: {str(e)}")

        flash(f'Successfully upgraded to {plan.title()} plan!', 'success')
    else:
        flash('Invalid plan selected', 'danger')

    return redirect(url_for('dashboard'))


@app.route('/admin/users')
@login_required
def admin_users():
    """Admin page to manage users (admin only)"""
    user = get_user_from_session()

    if not user or user.get('email') != ADMIN_EMAIL:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))

    # Get all users from Supabase
    users = []
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            users_data = supabase_request('GET', 'user_profiles?select=*,payments(*),plan_changes(*)')
            if users_data:
                users = users_data
        except Exception as e:
            logging.error(f"Error fetching users: {str(e)}")
            flash('Error fetching users from database', 'danger')

    return render_template('admin/users.html', users=users, user=user)


@app.route('/regenerate_api_key', methods=['POST'])
@login_required
def regenerate_api_key():
    """Regenerate user's API key"""
    user = get_user_from_session()
    plan = user.get('plan', 'free')
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS['free'])

    if not limits['api_access']:
        return jsonify({'error': 'API access requires Premium plan'}), 403

    try:
        new_api_key = generate_api_key()

        # Store in session as fallback if Supabase isn't available
        session[f'api_key_{user["id"]}'] = new_api_key

        if SUPABASE_URL and SUPABASE_KEY:
            try:
                update_data = {
                    'api_key': new_api_key,
                    'updated_at': datetime.now().isoformat()
                }

                response = supabase_request('PATCH', f'user_profiles?id=eq.{user["id"]}', update_data)

                if response:
                    logging.info(f"API key regenerated in Supabase for user {user['id']}")
                else:
                    logging.warning(f"Failed to update API key in Supabase, using session fallback")

            except Exception as e:
                logging.error(f"Error updating API key in Supabase: {str(e)}")
                logging.info("Using session-based API key storage as fallback")

        return jsonify({'api_key': new_api_key, 'success': True})

    except Exception as e:
        logging.error(f"Error regenerating API key: {str(e)}")
        return jsonify({'error': f'Failed to regenerate API key: {str(e)}'}), 500


@app.route('/history')
@login_required
def view_history():
    """View user's search history"""
    user = get_user_from_session()
    history = get_search_history(user['id'])

    return render_template('history.html', user=user, history=history)


@app.route('/history/<int:history_id>')
@login_required
def view_history_detail(history_id):
    """View detailed report from history"""
    user = get_user_from_session()

    # Get specific history item
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            history_item = supabase_request('GET', f'search_history?id=eq.{history_id}&user_id=eq.{user["id"]}')
            if history_item and len(history_item) > 0:
                item = history_item[0]

                return render_template('tool.html',
                                       keywords=item.get('keywords', []),
                                       url=item.get('url', ''),
                                       seo_suggestions=item.get('seo_suggestions', []),
                                       audit_results=[],
                                       seo_data=item.get('seo_data', {}),
                                       user=user,
                                       usage=get_user_usage(user['id']),
                                       limits=PLAN_LIMITS.get(user.get('plan', 'free'), PLAN_LIMITS['free']),
                                       user_plan=user.get('plan', 'free'),
                                       from_history=True)
        except Exception as e:
            logging.error(f"Error getting history detail: {str(e)}")

    # Fallback to session history
    history = session.get('search_history', [])
    if history_id < len(history):
        item = history[history_id]
        return render_template('tool.html',
                               keywords=item.get('keywords', []),
                               url=item.get('url', ''),
                               seo_suggestions=item.get('seo_suggestions', []),
                               audit_results=[],
                               seo_data=item.get('seo_data', {}),
                               user=user,
                               usage=get_user_usage(user['id']),
                               limits=PLAN_LIMITS.get(user.get('plan', 'free'), PLAN_LIMITS['free']),
                               user_plan=user.get('plan', 'free'),
                               from_history=True)

    flash('History item not found', 'error')
    return redirect(url_for('view_history'))


@app.route('/admin/change_plan/<user_id>/<new_plan>')
@login_required
def admin_change_plan(user_id, new_plan):
    """Admin function to change user plan"""
    user = get_user_from_session()

    if not user or user.get('email') != ADMIN_EMAIL:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))

    if new_plan not in ['free', 'pro', 'premium']:
        flash('Invalid plan selected', 'danger')
        return redirect(url_for('admin_users'))

    # Update plan in Supabase
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            # Update user plan directly
            update_data = {
                'plan': new_plan,
                'updated_at': datetime.now().isoformat()
            }

            response = supabase_request('PATCH', f'user_profiles?id=eq.{user_id}', update_data)

            if response:
                # Log admin action
                admin_action = {
                    'admin_id': user['id'],
                    'action_type': 'plan_change',
                    'target_user_id': user_id,
                    'details': {'new_plan': new_plan, 'changed_by': 'admin'}
                }
                supabase_request('POST', 'admin_actions', admin_action)

                flash(f'User plan successfully changed to {new_plan.title()}', 'success')
            else:
                flash('Error updating user plan', 'danger')

        except Exception as e:
            logging.error(f"Error changing user plan: {str(e)}")
            flash('Error updating user plan', 'danger')

    return redirect(url_for('admin_users'))


@app.route('/webhook/stripe', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhooks for automatic plan upgrades"""
    try:
        payload = request.get_data()
        sig_header = request.headers.get('Stripe-Signature')

        # In production, verify the webhook signature
        # event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)

        # For demo, parse the JSON directly
        event_data = request.get_json()

        if event_data.get('type') == 'checkout.session.completed':
            session_data = event_data.get('data', {}).get('object', {})
            customer_email = session_data.get('customer_details', {}).get('email')

            # Determine plan from amount or metadata
            amount = session_data.get('amount_total', 0)
            if amount == 1000:  # $10.00
                plan = 'pro'
            elif amount == 2000:  # $20.00
                plan = 'premium'
            else:
                logging.error(f"Unknown payment amount: {amount}")
                return jsonify({'status': 'error'}), 400

            # Upgrade user plan
            if customer_email and SUPABASE_URL and SUPABASE_KEY:
                try:
                    upgrade_data = {
                        'user_email': customer_email,
                        'new_plan': plan,
                        'stripe_subscription_id': session_data.get('subscription'),
                        'period_end': (datetime.now() + timedelta(days=30)).isoformat()
                    }

                    response = supabase_request('POST', 'rpc/upgrade_user_plan', upgrade_data)

                    if response:
                        # Log payment
                        payment_data = {
                            'stripe_payment_intent_id': session_data.get('payment_intent'),
                            'amount': amount,
                            'status': 'paid',
                            'plan_purchased': plan,
                            'processed_at': datetime.now().isoformat()
                        }

                        # Get user ID first
                        user_data = supabase_request('GET', f'user_profiles?email=eq.{customer_email}&select=id')
                        if user_data and len(user_data) > 0:
                            payment_data['user_id'] = user_data[0]['id']
                            supabase_request('POST', 'payments', payment_data)

                        logging.info(f"Successfully upgraded {customer_email} to {plan}")
                    else:
                        logging.error(f"Failed to upgrade {customer_email} to {plan}")

                except Exception as e:
                    logging.error(f"Error processing webhook: {str(e)}")
                    return jsonify({'status': 'error'}), 500

        return jsonify({'status': 'success'}), 200

    except Exception as e:
        logging.error(f"Webhook error: {str(e)}")
        return jsonify({'status': 'error'}), 400

# Add API endpoints for training data
@app.route('/api/training_data', methods=['GET'])
@api_key_required
def get_training_data():
    """API endpoint to retrieve training data."""
    if not TRAINER_AVAILABLE:
        return jsonify({'error': 'Training system unavailable'}), 500

    trainer = get_keyword_trainer()
    data = trainer.get_training_data()
    return jsonify(data)


@app.route('/api/training_data', methods=['POST'])
@api_key_required
def add_training_data():
    """API endpoint to add new training data."""
    if not TRAINER_AVAILABLE:
        return jsonify({'error': 'Training system unavailable'}), 500

    data = request.get_json()
    if not data or 'keyword' not in data or 'label' not in data:
        return jsonify({'error': 'Keyword and label are required'}), 400

    keyword = data['keyword']
    label = data['label']

    trainer = get_keyword_trainer()
    try:
        trainer.add_training_data(keyword, label)
        trainer.save_training_data()  # Persist the changes
        return jsonify({'message': 'Training data added successfully'}), 201
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/retrain_model', methods=['POST'])
@api_key_required
def retrain_model():
    """API endpoint to manually trigger model retraining."""
    if not TRAINER_AVAILABLE:
        return jsonify({'error': 'Training system unavailable'}), 500

    trainer = get_keyword_trainer()
    try:
        trainer.retrain_model()
        return jsonify({'message': 'Model retraining triggered'}), 200
    except Exception as e:
        logging.error(f"Error retraining model: {str(e)}")
        return jsonify({'error': f'Model retraining failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)