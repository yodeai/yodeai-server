web: gunicorn app:app -k uvicorn.workers.UvicornWorker --timeout 120 --preload --max-requests 1200
worker1: celery -A app.celery worker -n w1 --loglevel=info -Q processBlock --concurrency=1 --without-heartbeat --without-gossip --without-mingle
worker2: celery -A app.celery worker -n w2 --loglevel=info -Q processAncestors --concurrency=1 --without-heartbeat --without-gossip --without-mingle
