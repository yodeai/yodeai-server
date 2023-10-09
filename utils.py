
import time
import random
import sys
import os
from openai import ChatCompletion
#from langchain.vectorstores import SupabaseVectorStore
#from langchain.embeddings import HuggingFaceEmbeddings
import re
import requests
import json
import openai
from DB import supabaseClient
import boto3


from dotenv import load_dotenv
load_dotenv(dotenv_path='.env.local')
openai.api_key = os.getenv("OPENAI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')


s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

def exponential_backoff(retries=5, backoff_in_seconds=1, out=sys.stdout):
    def backoff(func):
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except:
                    if x == retries:
                        out.write(f"exception raised; number of retries is over {retries}\n")
                        raise
                    sleep_duration = (backoff_in_seconds * 2**x + random.uniform(0, 1))
                    out.write(f"exception raised; sleeping for: {sleep_duration} seconds\n")
                    time.sleep(sleep_duration)
                    x += 1
        return wrapper
    return backoff

@exponential_backoff(retries=5, backoff_in_seconds=1, out=sys.stdout)
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]



def addHyperlinksToResponse(response, linkMap):
    keyList = list(linkMap.keys())
    keyList.sort(key=lambda a: len(a))
    newResponse = response
    i = 0
    while i < len(newResponse):
        for key in keyList:
            if newResponse[i:].startswith(key):
                href = f"[{key}]({linkMap[key]})"
                newResponse = f"{newResponse[0: i]}{href}{newResponse[i + len(key):]}"
                i += len(href) - 1
                break
        i += 1
    return newResponse

def match(lst, metadataObj):
    return any([meta for meta in lst if "title" in meta and "source" in meta and "language" in meta and meta["title"] == metadataObj.get("title", None) and meta["source"] == metadataObj.get("source", None) and meta["language"] == metadataObj.get("language", None)])

def removeDuplicates(metadataList):
    uniqueList = []
    for meta in metadataList:
        if not match(uniqueList, meta):
            uniqueList.append(meta)
    return uniqueList



def getRelevance(question, response, text):
    myprompt = f"Is this information  ''{response}'' relevant to part of the following text in triple quotes? Answer with a single number between 1 to 10 with higher numbers representing higher relevance. Text:  '''{text}'''"
    resp2prompt = get_completion(myprompt)
    if resp2prompt==None:
        return 0
    else:
        score = re.match(r"\d+", resp2prompt)
        return int(score[0]) if score else 0

def getDocumentsVectorStore():
    client = supabaseClient
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return SupabaseVectorStore(
        client=client,
        embedding=embeddings,
        table_name="documents_huggingface",
        query_name="match_documents_huggingface"
    )
def getQuestionsVectorStore():
    client = supabaseClient
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return SupabaseVectorStore(
        client=client,
        embedding=embeddings,
        table_name="questions_huggingface",
        query_name="match_questions_huggingface"
    )


def remove_invalid_surrogates(text):
    # Remove invalid high surrogates
    text = re.sub(r'[\uD800-\uDBFF](?![\uDC00-\uDFFF])', '', text)
    # Remove invalid low surrogates
    text = re.sub(r'(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]', '', text)
    return text

# Current options for model are "BGELARGE_MODEL" and "MINILM_MODEL"
def getEmbeddings(texts, model='BGELARGE_MODEL'):
    
    headers = {
        "Authorization": f"Bearer {os.environ.get('HUGGINGFACEHUB_API_KEY')}",
        "Content-Type": "application/json"         
    }
    data = {"inputs": texts}
    
    #print(cleaned_texts[36:37])
    response = requests.post(os.environ.get(model), headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        print("Error in Hugging Face API Response:", response.content.decode("utf-8"))
        return None

    embeddings = json.loads(response.content.decode("utf-8"))
    if embeddings is None or 'embeddings' not in embeddings:
        raise Exception("Error in getting embeddings.")
    return embeddings['embeddings']

