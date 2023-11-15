from celery import shared_task

from processBlock import processBlock  
from fastapi import HTTPException
from utils import supabaseClient
from datetime import datetime


@shared_task(name='processBlock:process_a_block_task', bind=True,autoretry_for=(Exception,), retry_jitter=True, retry_backoff=5, retry_kwargs={"max_retries": 5}, task_ignore_result = True)
def process_block_task(self, block_id):
    try:
        processBlock(block_id)
        now = datetime.now()
        update_response, update_error = supabaseClient.table('processBlockLogging')\
            .insert({"block_id": block_id, "time": now.isoformat()}) \
            .execute()
        return {"block_id": block_id, "status": "Content processed and chunks stored successfully."}
    except Exception as e:
        update_response, update_error = supabaseClient.table('block')\
            .update({'status': 'failure'})\
            .eq('block_id', block_id)\
            .execute()
        raise HTTPException(status_code=400, detail=str(e))