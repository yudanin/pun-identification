"""
WSGI entry point for PIE - Pun Identification Engine

For PythonAnywhere deployment:
1. Set this as the WSGI file path in the Web tab
2. Update the path below to match your installation
3. Set ANTHROPIC_API_KEY in the virtualenv section
"""

import sys
import os

# --- 1. Set Project Root Path ---
# This must point to the folder containing the 'pie' package and 'claudeapikey'
project_home = '/home/areteai/PunID'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# --- 2. Add User Site-Packages Path (NO VENV) ---
# This is mandatory since you are not using a virtual environment and packages
# like 'anthropic' and 'spacy' are installed here.
# NOTE: The Python version is assumed to be 3.10 based on your settings.
user_site_packages = '/home/areteai/.local/lib/python3.10/site-packages'
if user_site_packages not in sys.path:
    sys.path.insert(0, user_site_packages)

# --- 3. Load the API Key from File (Engine should read it, but this is a fail-safe) ---
# This forces the key into the environment for the app, which your engine checks as a fallback.
api_key_path = os.path.join(project_home, 'claudeapikey')
if os.path.exists(api_key_path):
    # Reads the key from the file and sets it as an environment variable for this app only
    os.environ['ANTHROPIC_API_KEY'] = open(api_key_path).read().strip()

# --- 4. Import the Flask App (Assuming flattened 'pie' structure) ---
# Your api.py imports the engine, which starts the key-reading process.
from pie.api import create_app
application = create_app()
