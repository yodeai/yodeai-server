from celery import shared_task

from processBlock import processBlock
from processAncestors import processAncestors
from fastapi import HTTPException
from utils import supabaseClient
from datetime import datetime
from competitive_analysis import create_competitive_analysis, update_whiteboard_status, update_whiteboard_nodes
from user_analysis import generate_user_analysis
from painpoint_analysis import cluster_reviews, update_spreadsheet_nodes, update_spreadsheet_status
from review_scraper import App_Store_Scraper
from create_jira_tickets import create_jira_tickets, update_widget_status, update_widget_nodes

@shared_task(name='jiraGeneration:jira_generation_task', bind=True,autoretry_for=(Exception,), retry_jitter=True, retry_backoff=5, retry_kwargs={"max_retries": 1}, task_ignore_result = True)
def jira_generation_task(self, widget_id, prd_block_id, fields):
    try:
        output_data = create_jira_tickets(widget_id, prd_block_id, -1, fields)
        update_widget_nodes(output_data, widget_id)
        update_widget_status("success", widget_id)
        return {"widget_id": widget_id, "status": "jira generation done"}
    except Exception as e:
        print("Exception")
        update_widget_status("error", widget_id)
        raise HTTPException(status_code=400, detail=str(e))
    
@shared_task(name='processAncestors:painpoint_analysis_task', bind=True,autoretry_for=(Exception,), retry_jitter=True, retry_backoff=5, retry_kwargs={"max_retries": 1}, task_ignore_result = True)
def painpoint_analysis_task(self, owner_id, topics, lens_id, spreadsheet_id, num_clusters, app_name=""):
    try:
        if app_name:
            scraper_instance = App_Store_Scraper("us", app_name)
            scraper_instance.review(num_pages=10, max_rating=1, after=None, sleep=1)
            scraper_instance.add_to_lens(owner_id, lens_id)
        output_data = cluster_reviews(lens_id, topics, spreadsheet_id, num_clusters)
        update_spreadsheet_status("success", spreadsheet_id)
        update_spreadsheet_nodes(output_data, spreadsheet_id)
        return {"whiteboard_id": spreadsheet_id, "status": "painpoint analysis done"}
    except Exception as e:
        print("Exception")
        update_spreadsheet_status("error", spreadsheet_id)
        raise HTTPException(status_code=400, detail=str(e))
    

@shared_task(name='userAnalysis:user_analysis_task', bind=True,autoretry_for=(Exception,), retry_jitter=True, retry_backoff=5, retry_kwargs={"max_retries": 1}, task_ignore_result = True)
def user_analysis_task(self, topics, lens_id, whiteboard_id, block_ids=[]):
    try:
        output_data = generate_user_analysis(topics, lens_id, whiteboard_id, block_ids)
        update_whiteboard_status("success", whiteboard_id)
        update_whiteboard_nodes(output_data, whiteboard_id)
        return {"whiteboard_id": whiteboard_id, "status": "user analysis done"}
    except Exception as e:
        print("Exception")
        update_whiteboard_status("error", whiteboard_id)
        raise HTTPException(status_code=400, detail=str(e))
    

@shared_task(name='competitiveAnalysis:competitive_analysis_task', bind=True,autoretry_for=(Exception,), retry_jitter=True, retry_backoff=5, retry_kwargs={"max_retries": 1}, task_ignore_result = True)
def competitive_analysis_task(self, company_mapping, areas, whiteboard_id):
    try:
        print("hi", company_mapping)
        for key, value in company_mapping.items():
            if value:
                company_mapping[key] = value.replace("%20", " ")
        output_data = create_competitive_analysis(company_mapping, areas, whiteboard_id)
        update_whiteboard_status("success", whiteboard_id)
        update_whiteboard_nodes(output_data, whiteboard_id)
        return {"whiteboard_id": whiteboard_id, "status": "competitive analysis done"}
    except Exception as e:
        print("Exception")
        update_whiteboard_status("error", whiteboard_id)
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