"""
Production WSGI Entrypoint for AI College Study Assistant.
Can be run with:
    waitress-serve --listen=127.0.0.1:5000 wsgi:app
or:
    gunicorn -w 4 -b 127.0.0.1:5000 wsgi:app
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
    print("Serving production application with Waitress on http://127.0.0.1:5000 ...")
    serve(app, host="127.0.0.1", port=5000)
