#!/usr/bin/env python
from flask import Flask, send_from_directory
import os

# Create the Flask application, pointing static_folder to 'static'
app = Flask(__name__, static_folder='static')

# Route for the root URL: serve index.html
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# Route for all other GET requests: serve static files
@app.route('/<path:filename>')
def static_files(filename):
    # Ensure the file exists to return 404 otherwise
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    # Run the app on localhost:5000
    app.run(host='0.0.0.0', port=5000, debug=True)
