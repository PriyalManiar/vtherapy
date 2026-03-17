#!/usr/bin/env python3
"""
VTherapy app launcher. Run from project root: python run.py
In Docker, set FLASK_RUN_HOST=0.0.0.0 so the app is reachable from outside the container.
"""
import os
from scripts.mainapp import app, db

if __name__ == "__main__":
    host = os.environ.get("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_RUN_PORT", "5001"))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() in ("1", "true", "yes")
    print("Starting Flask app...")
    with app.app_context():
        db.create_all()
    app.run(host=host, port=port, debug=debug)
