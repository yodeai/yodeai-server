import os
import requests
import json
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env.local')
from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
from answerQuestion import answer_question 
from answerQuestionLens import answer_question_lens     
from pydantic import BaseModel

app = FastAPI()
class Question(BaseModel):
    text: str

class QuestionFromLens(BaseModel):
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


@app.get("/ask", response_class=HTMLResponse)
async def ask_form(request: Request):
    return templates.TemplateResponse("ask.html", {"request": request})

@app.get("/asklens", response_class=HTMLResponse)
async def ask_form(request: Request):
    return templates.TemplateResponse("asklens.html", {"request": request})

@app.post("/answer")
async def answer_text(question: str = Form(...)):
    answer = answer_question(question)
    return {"answer": answer}



@app.post("/answerFromLens")
async def answer_from_lens(data: QuestionFromLens):
    # Extracting question and lensID from the request body
    question = data.question
    lensID = data.lensID
    response = answer_question_lens(question, lensID)
    return response


@app.get("/test") # this is testing Hugging Face API for embeddings
def demo():
    # Set up the request headers and data
    headers = {
    "Authorization": f"Bearer {os.environ.get('HUGGINGFACEHUB_API_KEY')}",
    "Content-Type": "application/json"         
    }

    data = {"inputs": ["This is a sentence.", "This is another sentence.", "this is a sentence about Japanese food", "Sushi is nice"]}

    # Send the request to the Hugging Face API
    response = requests.post("https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5", headers=headers, data=json.dumps(data))
    #print(response.content)

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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)