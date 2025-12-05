"""
Flask API for PIE - Pun Identification Engine

Provides REST endpoints for pun analysis.
"""

import os
import logging
from flask import Flask, request, jsonify
from functools import wraps

from pie import PunIdentificationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize engine (API key can be set later)
engine = None


def get_engine():
    """Get or create the PIE engine."""
    global engine
    if engine is None:
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if api_key:
            engine = PunIdentificationEngine(api_key=api_key)
            logger.info("PIE engine initialized with API key from environment")
        else:
            engine = PunIdentificationEngine()
            logger.warning("PIE engine created without API key")
    return engine


def require_configured(f):
    """Decorator to ensure engine is configured before handling request."""
    @wraps(f)
    def decorated(*args, **kwargs):
        eng = get_engine()
        if not eng.is_configured:
            return jsonify({
                "error": "Engine not configured",
                "message": "Set ANTHROPIC_API_KEY environment variable or call /configure endpoint"
            }), 503
        return f(*args, **kwargs)
    return decorated


# =============================================================================
# API Endpoints
# =============================================================================

@app.route('/', methods=['GET'])
def index():
    """API information endpoint."""
    return jsonify({
        "name": "PIE - Pun Identification Engine",
        "version": "0.1.0",
        "endpoints": {
            "GET /": "This information",
            "GET /status": "Engine status",
            "POST /configure": "Configure API key",
            "POST /analyze": "Analyze a sentence for puns",
            "POST /analyze/batch": "Analyze multiple sentences"
        }
    })


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
    return jsonify({"error": "Not found", "message": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal error", "message": str(e)}), 500


# =============================================================================
# Main Entry Point
# =============================================================================

def create_app():
    """Factory function for creating the Flask app."""
    return app


if __name__ == '__main__':
    # Development server
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug)
