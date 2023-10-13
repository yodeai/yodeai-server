web: gunicorn app:app -k uvicorn.workers.UvicornWorker --timeout 120

worker: celery -A app.celery worker --loglevel=info -Q processBlock -E

flower: celery -A yodeai-server flower
