import os
import requests
import json
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env.local')
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
from answerQuestionLens import answer_question_lens, get_searchable_feed, update_question_popularity 
from celery_tasks.tasks import process_block_task 
from pydantic import BaseModel
from config.celery_utils import create_celery
from config.celery_utils import get_task_info
from celery.signals import task_success, task_failure, task_internal_error
from utils import exponential_backoff, supabaseClient
import sys
def create_app() -> FastAPI:
    current_app = FastAPI()
    current_app.celery_app = create_celery()
    return current_app

app = create_app()
celery = app.celery_app
class Question(BaseModel):
    text: str

class QuestionFromLens(BaseModel):
    question: str
    lens_id: str

class QuestionPopularityUpdateFromLens(BaseModel):
    row_id: str
    lens_id: str

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for better security later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.get("/task/{task_id}")
async def get_task_status(task_id: str) -> dict:
    """
    Return the status of the submitted Task
    """
    return get_task_info(task_id)

@app.get("/asklens", response_class=HTMLResponse)
async def ask_form(request: Request):
    return templates.TemplateResponse("asklens.html", {"request": request})


@app.post("/searchableFeed")
async def searchable_feed(data: QuestionFromLens):
    # Perform your logic here to get the answer
    answer = get_searchable_feed(data.question, data.lens_id)
    return {"answer": answer}

@app.patch("/increasePopularity")
async def answer_text(data: QuestionPopularityUpdateFromLens):
    # Perform your logic here to get the answer
    answer = update_question_popularity(data.row_id, 1, data.lens_id)
    return {"answer": answer}

@app.patch("/decreasePopularity")
async def answer_text(data: QuestionPopularityUpdateFromLens):
    # Perform your logic here to get the answer
    answer = update_question_popularity(data.row_id, -1, data.lens_id)
    return {"answer": answer}

@app.post("/processBlock")
async def route_process_block(block: dict):
    block_id = block.get("block_id")
    if not block_id:
        raise HTTPException(status_code=400, detail="block_id must be provided")
    task = process_block_task.apply_async(args=[block_id])
    return JSONResponse({"task_id": task.id})

@app.post("/answerFromLens")
async def answer_from_lens(data: QuestionFromLens):
    # Extracting question and lens_id from the request body
    question = data.question
    lens_id = data.lens_id
    response = answer_question_lens(question, lens_id)
    return response


@app.get("/test") # this is testing Hugging Face API for embeddings
def demo():
    # Set up the request headers and data
    headers = {
    "Authorization": f"Bearer {os.environ.get('HUGGINGFACEHUB_API_KEY')}",
    "Content-Type": "application/json"         
    }


    data = {"wait_for_model": True, "inputs": ["This is a sentence.", "This is another sentence.", "this is a sentence about Japanese food", "Sushi is nice"]}

    # Send the request to the Hugging Face API
    @exponential_backoff(retries=5, backoff_in_seconds=1, out=sys.stdout)
    def get_response(headers, data):
        response = requests.post(os.environ.get('BGELARGE_MODEL'), headers=headers, data=json.dumps(data))
        return response
    response = get_response(headers, data)

    if response.status_code != 200:
        print("Error in Hugging Face API Response:", response.content.decode("utf-8"))
        return {"Error in API response": response.content.decode("utf-8")}

    response_content = response.content.decode("utf-8")
    print("Hugging Face API Response:", response_content)
    

    # Extract the embeddings from the response
    embeddings = json.loads(response.content.decode("utf-8"))
    #print(embeddings[0])
    

    # Calculate the pairwise similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    

    # Print the similarity matrix
    for i, row in enumerate(similarity_matrix):
        print(f"Sentence {i+1} similarity: {row}")

    return {"similarity_matrix": similarity_matrix.tolist()}

@task_success.connect
def task_success_notifier(sender=None, result=None, **kwargs):
    # After processing all chunks, update the status of the block to 'ready'
    print("updating block", result['block_id'])
    update_response, update_error = supabaseClient.table('block')\
        .update({'status': 'ready'})\
        .eq('block_id', result['block_id'])\
        .execute()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)