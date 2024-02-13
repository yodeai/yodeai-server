from celery import shared_task

from processBlock import processBlock
from processAncestors import processAncestors
from fastapi import HTTPException
from utils import supabaseClient
from datetime import datetime
from competitive_analysis import create_competitive_analysis, update_whiteboard_status, update_whiteboard_nodes
from user_analysis import generate_user_analysis
from painpoint_analysis import cluster_reviews, update_spreadsheet_nodes, update_spreadsheet_status


@shared_task(name='processAncestors:painpoint_analysis_task', bind=True,autoretry_for=(Exception,), retry_jitter=True, retry_backoff=5, retry_kwargs={"max_retries": 1}, task_ignore_result = True)
def painpoint_analysis_task(self, topics, lens_id, spreadsheet_id, num_clusters):
    try:
        output_data = cluster_reviews(lens_id, topics, spreadsheet_id, num_clusters)
        update_spreadsheet_status("success", spreadsheet_id)
        update_spreadsheet_nodes(output_data, spreadsheet_id)
        return {"whiteboard_id": spreadsheet_id, "status": "painpoint analysis done"}
    except Exception as e:
        print("Exception")
        raise HTTPException(status_code=400, detail=str(e))
    

@shared_task(name='competitiveAnalysis:user_analysis_task', bind=True,autoretry_for=(Exception,), retry_jitter=True, retry_backoff=5, retry_kwargs={"max_retries": 1}, task_ignore_result = True)
def user_analysis_task(self, topics, lens_id, whiteboard_id):
    try:
        output_data = generate_user_analysis(topics, lens_id, whiteboard_id)
        update_whiteboard_status("success", whiteboard_id)
        update_whiteboard_nodes(output_data, whiteboard_id)
        return {"whiteboard_id": whiteboard_id, "status": "user analysis done"}
    except Exception as e:
        print("Exception")
        raise HTTPException(status_code=400, detail=str(e))
    

@shared_task(name='competitiveAnalysis:competitive_analysis_task', bind=True,autoretry_for=(Exception,), retry_jitter=True, retry_backoff=5, retry_kwargs={"max_retries": 1}, task_ignore_result = True)
def competitive_analysis_task(self, company_mapping, areas, whiteboard_id):
    try:
        output_data = create_competitive_analysis(company_mapping, areas, whiteboard_id)
        update_whiteboard_status("success", whiteboard_id)
        update_whiteboard_nodes(output_data, whiteboard_id)
        return {"whiteboard_id": whiteboard_id, "status": "competitive analysis done"}
    except Exception as e:
        print("Exception")
        raise HTTPException(status_code=400, detail=str(e))
    
@shared_task(name='processBlock:process_a_block_task', bind=True,autoretry_for=(Exception,), retry_jitter=True, retry_backoff=5, retry_kwargs={"max_retries": 1}, task_ignore_result = True)
def process_block_task(self, block_id):
    try:
        print("processing block")
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