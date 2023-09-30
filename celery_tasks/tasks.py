from typing import List

from celery import shared_task

from processBlock import processBlock  
from fastapi import HTTPException


@shared_task(name='processBlock:process_a_block_task', bind=True,autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5})
def process_block_task(self, block_id):
    try:
        processBlock(block_id)
        return {"status": "Content processed and chunks stored successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

