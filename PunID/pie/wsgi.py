"""
WSGI entry point for PIE - Pun Identification Engine

For PythonAnywhere deployment:
1. Set this as the WSGI file path in the Web tab
2. Update the path below to match your installation
3. Set ANTHROPIC_API_KEY in the virtualenv section
"""

import sys
import os

# Add your project path
# Update this path for your PythonAnywhere username
project_home = '/home/YOUR_USERNAME/pie'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Load environment variables from .env file if present
env_file = os.path.join(project_home, '.env')
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Import the Flask app
from api import create_app
application = create_app()
