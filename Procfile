web: gunicorn app:app -k uvicorn.workers.UvicornWorker --timeout 120

# Run Celery Worker
worker: celery -A app.celery worker --loglevel=info -Q processBlock -E

# Run Flower
    celery -A app.celery flower --port=5555