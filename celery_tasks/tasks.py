from celery import shared_task

from processBlock import processBlock
from processAncestors import processAncestors
from fastapi import HTTPException
from utils import supabaseClient
from datetime import datetime


@shared_task(name='processBlock:process_a_block_task', bind=True,autoretry_for=(Exception,), retry_jitter=True, retry_backoff=5, retry_kwargs={"max_retries": 1}, task_ignore_result = True)
def process_block_task(self, block_id):
    try:
        print("processing block")
        processBlock(block_id)
        print("done with that")
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

@shared_task(name='processAncestors:process_ancestors_task', bind=True,autoretry_for=(Exception,), retry_jitter=True, retry_backoff=5, retry_kwargs={"max_retries": 1}, task_ignore_result = True)
def process_ancestors_task(self, block_id, lens_id, remove):
    try:
        print("processing ancestors")
        processAncestors(block_id, lens_id, remove)
        print("Done")
        return {"block_id": block_id, "lens_id": lens_id, "status": "All ancestors deleted the block" if remove else "All ancestors added the block"}
    except Exception as e:
        print("Exception")
        raise HTTPException(status_code=400, detail=str(e))