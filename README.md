# Yodeai Server

This repository contains the back-end for Yodeai.

## Setup and Local Development

### Prerequisites

- Ensure you have a Python environment set up, preferably using `venv`. If you don't have a venv folder, create one: virtualenv venv

### Steps to Run Locally

1. **Activate the virtual environment:**
    ```bash
    source venv/bin/activate
    ```

2. **Install the necessary dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the FastAPI app:**
    ```bash
    uvicorn app:app --reload
    ```


## Deployment to Heroku

1. **Deploy to Heroku:**
    ```bash
    git push heroku main
    ```
