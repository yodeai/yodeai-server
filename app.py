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
from answerQuestionLens import answer_question_lens
from celery_tasks.tasks import process_block_task 
from pydantic import BaseModel
from config.celery_utils import create_celery
from config.celery_utils import get_task_info
from celery.signals import task_success
from utils import exponential_backoff, supabaseClient
import sys
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import EmailStr
from uuid import uuid4

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

class QuestionPopularityUpdateFromLens(BaseModel):
    row_id: str
    lensID: str
    activeComponent: str
    userID: str
    

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
    }
    data, count = supabaseClient.table('lens_invites').insert(insertData).execute()

    template = f"""
		<html>
		<body>
		<p>Hi {recipients[0]}!
        <br></br>
		<p>{sender} is inviting you to collaborate on the lens {lensId} with the role of: {role} </p>
        <p>Click <a href={inviteLink}>here</a> to accept the invite. </p>
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


# @app.post("/searchableFeed")
# async def searchable_feed(data: QuestionFromLens):
#     # Perform your logic here to get the answer
#     answer = get_searchable_feed(data.question, data.lensID)
#     return {"answer": answer}

# @app.patch("/increasePopularity")
# async def increase_popularity(data: QuestionPopularityUpdateFromLens):
#     # Perform your logic here to get the answer
#     answer = update_question_popularity(data.row_id, 1, data.lensID)
#     return {"answer": answer}

# @app.patch("/decreasePopularity")
# async def decrease_popularity(data: QuestionPopularityUpdateFromLens):
#     # Perform your logic here to get the answer
#     answer = update_question_popularity(data.row_id, -1, data.lensID)
#     return {"answer": answer}

# @app.post("/answer")
# async def answer_text(q: Question):
#     # Perform your logic here to get the answer
#     answer = answer_question(q.question)
#     return {"answer": answer}


@app.post("/processBlock")
async def route_process_block(block: dict):
    block_id = block.get("block_id")
    if not block_id:
        raise HTTPException(status_code=400, detail="block_id must be provided")
    task = process_block_task.apply_async(args=[block_id])
    return JSONResponse({"task_id": task.id})

@app.post("/answerFromLens")
async def answer_from_lens(data: QuestionFromLens):
    #return [data.lensID, type(data.lensID)]
    #sys.stdout.write("Debug message here\n")
    #sys.stdout.write(data.lensID+" "+data.activeComponent)
    question = data.question
    lensID = None if data.lensID == "NONE" else data.lensID
    userID = data.userID
    activeComponent = data.activeComponent
    response = answer_question_lens(question, lensID, activeComponent, userID)
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
