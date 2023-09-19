# YodeAI Server

This repository contains the back-end for YodeAI.

## Setup and Local Development

### Prerequisites

- Ensure you have a Python environment set up, preferably using `venv`.

### Steps to Run Locally

1. **Activate the virtual environment:**
    ```bash
    source venv/bin/activate
    ```

2. **Install the necessary dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set the Flask app environment variable:**
    ```bash
    export FLASK_APP=app.py 
    ```

4. **Run the Flask app:**
    ```bash
    flask run
    ```

## Deployment to Heroku

1. **Deploy to Heroku:**
    ```bash
    git push heroku main
    ```
