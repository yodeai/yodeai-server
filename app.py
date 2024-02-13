import os
import requests
import json
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env.local')
from fastapi import FastAPI, Request,  HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
from answerQuestionLens import answer_question_lens, get_searchable_feed, update_question_popularity
from celery_tasks.tasks import process_block_task, process_ancestors_task, competitive_analysis_task, user_analysis_task, painpoint_analysis_task
from pydantic import BaseModel
from config.celery_utils import create_celery
from config.celery_utils import get_task_info
from celery.signals import task_success
from utils import exponential_backoff, supabaseClient
import sys
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import EmailStr
from uuid import uuid4
from typing import Optional
import asyncio
import httpx
from competitive_analysis import update_whiteboard_status
from painpoint_analysis import update_spreadsheet_status
class LensInvite(BaseModel):
    sender: str
    lensId: str
    email: str
    role: str

class EmailSettings(BaseModel):
    email_server: str
    email_port: int
    email_username: str
    email_password: str
    sender_email: EmailStr
def create_app() -> FastAPI:
    current_app = FastAPI()
    current_app.celery_app = create_celery()
    return current_app
import sys

app = create_app()
celery = app.celery_app
class Question(BaseModel):
    text: str

class QuestionFromLens(BaseModel):
    question: str
    lensID: str
    userID: str
    activeComponent: str
    published: Optional[bool] = False
    google_user_id: Optional[str] = None

class QuestionPopularityUpdateFromLens(BaseModel):
    row_id: str
    lens_id: str
    user_id: str

class QuestionForSearchableFeed(BaseModel):
    question: str
    lensID: str

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

@app.post('/shareLens')
async def share_lens(sharing_details: dict):
    recipients = [sharing_details["email"]]
    lensId = sharing_details["lensId"]
    sender = sharing_details["sender"]
    role = sharing_details["role"]
    token = str(uuid4())
    inviteLink = f"{os.environ.get('BASE_URL')}/acceptInvite/{token}"
    insertData = {
        "sender": sender,
        "recipient": recipients[0],
        "token": token,
        "lens_id": lensId,
        "access_type": role,
        "status": "sent"
    }
    lensNameData, error = supabaseClient.table('lens').select('name').eq('lens_id', lensId).execute()
    lensName = lensNameData[1][0]['name']

    upsertData, upsertCount = supabaseClient.table('lens_invites').upsert(
        [insertData],
        on_conflict="sender, recipient, lens_id"
    ).execute()
    
    if upsertCount == 1:
        print("Updated sharing successfuly")
    else:
        print("Failed to update")
    template = f"""
        <html>
        <body>
        <p>Hi {recipients[0]}!
        <br></br>
        <p>{sender} is inviting you to collaborate on the space '{lensName}' with  ID {lensId}, offering you the role of: {role}.</p>
        <p>Click <a href={inviteLink}>here</a> to accept the invite for the space '{lensName}'. </p>
        </body>
        </html>
        """
    message = MessageSchema(
        subject=f"Yodeai: {sender} shared a lens with you!",
        recipients=recipients, # List of recipients, as many as you can pass 
        body=template,
        subtype="html"
        )

    # Example configuration
    email_config = EmailSettings(
        email_server="smtp.gmail.com",
        email_port=587,
        email_username=os.environ.get('EMAIL'),
        email_password=os.environ.get('APP_PASSWORD'),
        sender_email=os.environ.get('EMAIL'),
    )
    print(email_config)

    # Create a ConnectionConfig object
    conf = ConnectionConfig(
        MAIL_USERNAME=email_config.email_username,
        MAIL_PASSWORD=email_config.email_password,
        MAIL_SERVER=email_config.email_server,
        MAIL_PORT=email_config.email_port,
        MAIL_FROM=email_config.sender_email,
        MAIL_STARTTLS=True,  # Example value
        MAIL_SSL_TLS=False,  # Example value
    )
    fm = FastMail(conf)
    await fm.send_message(message)
    print(message)
    return JSONResponse(status_code=200, content={"message": "email has been sent"})


@app.post("/searchableFeed")
async def searchable_feed(data: QuestionForSearchableFeed):
    # Perform your logic here to get the answer
    answer = get_searchable_feed(data.question, data.lensID)
    return {"answer": answer}

@app.patch("/increasePopularity")
async def increase_popularity(data: QuestionPopularityUpdateFromLens):
    # Perform your logic here to get the answer
    answer = update_question_popularity(data.lens_id, data.user_id, data.row_id, 1)
    return {"answer": answer}

@app.patch("/decreasePopularity")
async def decrease_popularity(data: QuestionPopularityUpdateFromLens):
    # Perform your logic here to get the answer
    answer = update_question_popularity(data.lens_id, data.user_id, data.row_id, -1)
    return {"answer": answer}

# @app.post("/answer")
# async def answer_text(q: Question):
#     # Perform your logic here to get the answer
#     answer = answer_question(q.question)
#     return {"answer": answer}


# Store pending tasks
pending_tasks = {}
ongoing_tasks = {}

async def wait_and_send_task(block_id, countdown):
    await asyncio.sleep(countdown)
    # Check again before sending to see if the task is still pending
    if block_id in pending_tasks:
        # Send the task to the broker or execute it
        send_task_to_broker(block_id)
        # Remove the task from pending tasks
        del pending_tasks[block_id]


def send_task_to_broker(block_id):
    # Replace this with the logic to send the task to the broker
    print(f"Sending process block task for block_id {block_id} to the broker")
    if block_id in ongoing_tasks:
        ongoing_tasks[block_id].revoke(terminate=True)
    task = process_block_task.apply_async(args=[block_id])
    ongoing_tasks[block_id] = task
    return JSONResponse({"task_id": task.id})

def schedule_task(block_id, countdown):
    # Check if there's already a pending task for the same block_id
    if block_id in pending_tasks:
        # Cancel the existing pending task (optional, based on your logic)
        del pending_tasks[block_id]

    # Store the new task with its countdown
    pending_tasks[block_id] = {"countdown": countdown}

    # Check again before starting the countdown
    if block_id in pending_tasks:
        # Start the countdown mechanism
        asyncio.create_task(wait_and_send_task(block_id, countdown))

@app.post("/processBlock")
async def route_process_block(block: dict):
    block_id = block.get("block_id")
    countdown = block.get("delay", 0)
    if not block_id:
        raise HTTPException(status_code=400, detail="block_id must be provided")
    schedule_task(block_id, countdown)

async def exchange_code_for_google_tokens(code: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                "redirect_uri": "http://localhost:3000/auth",
                "grant_type": "authorization_code",
            },
        )
        return response.json()
    
@app.post("/googleAuth")
async def google_auth(body: dict):
    code = body.get("code")
    google_tokens = await exchange_code_for_google_tokens(code)

    # You can now store google_tokens in your database or use it as needed
    return {"google_tokens": google_tokens}

def update_whiteboard_task_id(task_id, whiteboard_id):
    # Update the status of the block
    update_response, update_error = supabaseClient.table('whiteboard')\
        .update({'task_id': task_id})\
        .eq('whiteboard_id', whiteboard_id)\
        .execute()
    

def update_spreadsheet_task_id(task_id, spreadsheet_id):
    # Update the status of the block
    update_response, update_error = supabaseClient.table('spreadsheet')\
        .update({'task_id': task_id})\
        .eq('spreadsheet_id', spreadsheet_id)\
        .execute()

@app.post("/competitiveAnalysis")
async def route_competitive_analysis(data: dict):
    company_mapping = data.get("company_mapping")
    whiteboard_id = data.get("whiteboard_id")
    areas = data.get("areas")
    # Get the plugin
    if not whiteboard_id or not company_mapping or not areas:
        raise HTTPException(status_code=400, detail="whiteboard id, urls, and areas must be provided")
    update_whiteboard_status("queued", whiteboard_id)
    task = competitive_analysis_task.apply_async(args=[company_mapping, areas, whiteboard_id])
    update_whiteboard_task_id(task.id, whiteboard_id)
    return JSONResponse({"task_id": task.id, "type": "competitive_analysis"})

@app.post("/userAnalysis")
async def route_user_analysis(data: dict):
    topics = data.get("topics")
    whiteboard_id = data.get("whiteboard_id")
    lens_id = data.get("lens_id")
    # Get the plugin
    update_whiteboard_status("queued", whiteboard_id)
    task = user_analysis_task.apply_async(args=[topics, lens_id, whiteboard_id])
    update_whiteboard_task_id(task.id, whiteboard_id)
    return JSONResponse({"task_id": task.id, "type": "user_analysis"})


@app.post("/painpointAnalysis")
async def route_user_analysis(data: dict):
    topics = data.get("painpoints")
    spreadsheet_id = data.get("spreadsheet_id")
    lens_id = data.get("lens_id")
    num_clusters = data.get("num_clusters")
    # Get the plugin
    update_spreadsheet_status("queued", spreadsheet_id)
    task = painpoint_analysis_task.apply_async(args=[topics, lens_id, spreadsheet_id, num_clusters])
    update_spreadsheet_task_id(task.id, spreadsheet_id)
    return JSONResponse({"task_id": task.id, "type": "painpoint_analysis"})



@app.post("/revokeTask")
async def route_user_analysis(data: dict):
    task_id = data.get("task_id")
    if task_id:
        # Revoke the task
        celery.control.revoke(task_id, terminate=True)
        print("Revoked!")
        return JSONResponse({'status': 'Task revoked successfully'})
    else:
        print("Error revoking")
        return JSONResponse({'error': 'Task ID not provided'})
    
@app.post("/processAncestors")
async def route_process_ancestors(information: dict):
    block_id = information.get("block_id")
    lens_id = information.get("lens_id")
    remove = information.get("remove")
    if not block_id or not lens_id:
        raise HTTPException(status_code=400, detail="block_id/lens_id must be provided")
    print(f"Sending ancestors task for block_id {block_id} to the broker")
    task = process_ancestors_task.apply_async(args=[block_id, lens_id, remove])
    return JSONResponse({"task_id": task.id, "type": "process_ancestors"})

@app.post("/answerFromLens")
async def answer_from_lens(data: QuestionFromLens):
    #return [data.lensID, type(data.lensID)]
    #sys.stdout.write("Debug message here\n")
    #sys.stdout.write(data.lensID+" "+data.activeComponent)
    question = data.question
    lensID = None if data.lensID == "NONE" else data.lensID
    userID = data.userID
    activeComponent = data.activeComponent
    published = data.published
    google_user_id = None if data.google_user_id == "NONE" else data.google_user_id
    response = answer_question_lens(question, lensID, activeComponent, userID, published, google_user_id)
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

    

    # Calculate the pairwise similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    

    # Print the similarity matrix
    for i, row in enumerate(similarity_matrix):
        print(f"Sentence {i+1} similarity: {row}")

    return {"similarity_matrix": similarity_matrix.tolist()}

# @task_success.connect
# def task_success_notifier(sender=None, result=None, **kwargs):
#     task_name = result.get('task_name')
#     # After processing all chunks, update the status of the block to 'ready'
#     print("updating block", result['block_id'])
#     update_response, update_error = supabaseClient.table('block')\
#         .update({'status': 'ready'})\
#         .eq('block_id', result['block_id'])\
#         .execute()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)    

