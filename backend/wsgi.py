"""
Production WSGI Entrypoint for AI College Study Assistant.
Supports Waitress, Gunicorn, and cloud deployment on Render / Railway.
"""

import os
import sys

# Ensure backend directory is in python search path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import create_app

app = create_app()

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0"
    print(f"Serving production application with Waitress on http://{host}:{port} ...")
    serve(app, host=host, port=port)
