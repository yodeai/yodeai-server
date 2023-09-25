
import time
import random
import sys
import requests
import json
import os
import openai
from supabase import create_client, Client
from openai import ChatCompletion
import re
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env.local')
openai.api_key = os.getenv("OPENAI_API_KEY")

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


def fetchLinksFromDatabase():
    url: str = os.environ.get("PUBLIC_SUPABASE_URL")
    key: str = os.environ.get("PUBLIC_SUPABASE_ANON_KEY")
    supabase  = create_client(url, key)
    data, count = supabase.table('links').select('title, url').execute()
    if len(data[1]) == 0:
        raise Exception("no data found in links database")
    data = data[1]
    linkMap = {}

    if data:
        for link in data:
            linkMap[link["title"]] = link["url"]

    return linkMap

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



def getRelevance(question, response, text):
    myprompt = f"Is this information  ''{response}'' relevant to part of the following text in triple quotes? Answer with a single number between 1 to 10 with higher numbers representing higher relevance. Text:  '''{text}'''"
    resp2prompt = get_completion(myprompt)
    if resp2prompt==None:
        return 0
    else:
        score = re.match(r"\d+", resp2prompt)
        return int(score[0]) if score else 0


def getEmbeddings(texts):
    headers = {
        "Authorization": f"Bearer {os.environ.get('HUGGINGFACEHUB_API_KEY')}",
        "Content-Type": "application/json"         
    }
    data = {"inputs": texts}

    response = requests.post("https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5", headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        print("Error in Hugging Face API Response:", response.content.decode("utf-8"))
        return None

    embeddings = json.loads(response.content.decode("utf-8"))
    return embeddings

