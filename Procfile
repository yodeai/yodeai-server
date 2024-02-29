web: gunicorn app:app -k uvicorn.workers.UvicornWorker --timeout 120 --preload --max-requests 1200
worker1: celery -A app.celery worker -n w1 --loglevel=info -Q processBlock --concurrency=1 --without-heartbeat --without-gossip --without-mingle
worker2: celery -A app.celery worker -n w2 --loglevel=info -Q processAncestors --concurrency=1 --without-heartbeat --without-gossip --without-mingle
worker3: celery -A app.celery worker -n w3 --loglevel=info -Q competitiveAnalysis --concurrency=1 --without-heartbeat --without-gossip --without-mingle
worker4: celery -A app.celery worker -n w4 --loglevel=info -Q userAnalysis --concurrency=1 --without-heartbeat --without-gossip --without-mingle
worker5: celery -A app.celery worker -n w5 --loglevel=info -Q jiraGeneration --concurrency=1 --without-heartbeat --without-gossip --without-mingle
