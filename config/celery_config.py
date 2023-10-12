import os
from functools import lru_cache
from kombu import Queue
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env.local')


def route_task(name, args, kwargs, options, task=None, **kw):
    if ":" in name:
        queue, _ = name.split(":")
        return {"queue": queue}
    return {"queue": "celery"}

class BaseConfig:
    CELERY_BROKER_URL: os.environ.get('CELERY_BROKER_URL')
    CELERY_RESULT_BACKEND: os.environ.get('CELERY_RESULT_BACKEND')

    CELERY_TASK_QUEUES: list = (
        # default queue
        Queue("celery"),
        # custom queue
        Queue("processBlock"),
    )

    CELERY_TASK_ROUTES = (route_task,)


class DevelopmentConfig(BaseConfig):
    pass


@lru_cache()
def get_settings():
    config_cls_dict = {
        "development": DevelopmentConfig,
    }
    config_name = os.environ.get("CELERY_CONFIG", "development")
    config_cls = config_cls_dict[config_name]
    return config_cls()


settings = get_settings()