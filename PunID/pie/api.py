"""
Flask API for PIE - Pun Identification Engine

Provides REST endpoints for pun analysis and serves the UI.
"""

import os
import logging
from flask import Flask, request, jsonify, render_template, redirect, url_for
from functools import wraps

# Assuming engine is correctly imported from the same package
from .engine import PunIdentificationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- GLOBAL PATHS (Determines location of UI files) ---

# 1. Define the base directory (PunID/) dynamically relative to the current file (PunID/pie/api.py)
# os.pardir moves up one level from 'pie' to 'PunID'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
UI_TEMPLATE_PATH = os.path.join(BASE_DIR, 'pui', 'templates')
UI_STATIC_PATH = os.path.join(BASE_DIR, 'pui', 'static')

# --- GLOBAL APP INSTANCE (Used by all decorators) ---

# 2. Instantiate the Flask app globally with custom paths
app = Flask(
    __name__,
    template_folder=UI_TEMPLATE_PATH,
    static_folder=UI_STATIC_PATH
)
app.config['JSON_SORT_KEYS'] = False

# Initialize engine globally
engine = None


# --- ENGINE MANAGEMENT ---

def get_engine():
    """Get or create the PIE engine."""
    global engine
    if engine is None:
        # NOTE: Relying on engine.__init__ to read the claudeapikey file
        engine = PunIdentificationEngine()
        if not engine.is_configured:
            logger.warning("PIE engine created but not configured (missing API key).")
        else:
            logger.info("PIE engine initialized successfully.")
    return engine


def require_configured(f):
    """Decorator to ensure engine is configured before handling request."""
    @wraps(f)
    def decorated(*args, **kwargs):
        eng = get_engine()
        if not eng.is_configured:
            return jsonify({
                "error": "Engine not configured",
                "message": "API key file not found or LLM client failed to initialize."
            }), 503
        return f(*args, **kwargs)
    return decorated


# =============================================================================
# UI Endpoints
# =============================================================================

@app.route('/', methods=['GET'])
def index_redirect():
    """Redirects root path to the UI."""
    return redirect(url_for('ui'))

@app.route('/pundemonium', methods=['GET'])
def ui():
    """Serves the main single-page application UI from /pui/templates/."""
    return render_template('pundemonium.html')

# =============================================================================
# API Endpoints
# =============================================================================

@app.route('/status', methods=['GET'])
def status():
    """Get engine status."""
    eng = get_engine()
    return jsonify({
        "status": "ready" if eng.is_configured else "not_configured",
        **eng.get_status()
    })


@app.route('/configure', methods=['POST'])
def configure():
    """Configure the engine with an API key."""
    data = request.get_json()

    if not data or 'api_key' not in data:
        return jsonify({
            "error": "Missing api_key",
            "message": "Provide 'api_key' in request body"
        }), 400

    try:
        eng = get_engine()
        eng.set_api_key(data['api_key'])
        return jsonify({
            "status": "configured",
            "message": "API key set successfully"
        })
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return jsonify({
            "error": "Configuration failed",
            "message": str(e)
        }), 500


@app.route('/analyze', methods=['POST'])
@require_configured
def analyze():
    """
    Analyze a sentence for puns.

    Request body:
    {
        "sentence": "The sentence to analyze"
    }

    Response:
    {
        "sentence": "...",
        "has_pun": 1 or 0,
        "puns": [...],
        "analysis_notes": "..."
    }
    """
    data = request.get_json()

    if not data or 'sentence' not in data:
        return jsonify({
            "error": "Missing sentence",
            "message": "Provide 'sentence' in request body"
        }), 400

    sentence = data['sentence'].strip()

    if not sentence:
        return jsonify({
            "error": "Empty sentence",
            "message": "Sentence cannot be empty"
        }), 400

    if len(sentence) > 5000:
        return jsonify({
            "error": "Sentence too long",
            "message": "Maximum sentence length is 5000 characters"
        }), 400

    try:
        eng = get_engine()
        result = eng.analyze(sentence)
        return jsonify(result.to_dict())
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({
            "error": "Analysis failed",
            "message": str(e)
        }), 500


@app.route('/analyze/batch', methods=['POST'])
@require_configured
def analyze_batch():
    """
    Analyze multiple sentences for puns.

    Request body:
    {
        "sentences": ["sentence1", "sentence2", ...]
    }

    Response:
    {
        "results": [
            {analysis result 1},
            {analysis result 2},
            ...
        ]
    }
    """
    data = request.get_json()

    if not data or 'sentences' not in data:
        return jsonify({
            "error": "Missing sentences",
            "message": "Provide 'sentences' array in request body"
        }), 400

    sentences = data['sentences']

    if not isinstance(sentences, list):
        return jsonify({
            "error": "Invalid format",
            "message": "'sentences' must be an array"
        }), 400

    if len(sentences) > 10:
        return jsonify({
            "error": "Too many sentences",
            "message": "Maximum 10 sentences per batch request"
        }), 400

    try:
        eng = get_engine()
        results = eng.analyze_batch(sentences)
        return jsonify({
            "results": [r.to_dict() for r in results]
        })
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        return jsonify({
            "error": "Batch analysis failed",
            "message": str(e)
        }), 500

# =============================================================================
# Error Handlers
# =============================================================================

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request", "message": str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    # This must be defined globally, not inside create_app
    return jsonify({"error": "Not found", "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal error", "message": str(e)}), 500


# =============================================================================
# Main Entry Point (WSGI Compatibility)
# =============================================================================

# This is the function the WSGI server calls, returning the globally configured app.
def create_app():
    return app


if __name__ == '__main__':
    # Development server
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

    # Must use the globally defined app
    app.run(host='0.0.0.0', port=port, debug=debug)