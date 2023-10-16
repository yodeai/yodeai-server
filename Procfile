web: gunicorn app:app -k uvicorn.workers.UvicornWorker --timeout 120 --preload --max-requests 1200
worker: celery -A app.celery worker --loglevel=info -Q processBlock -E --concurrency=1
