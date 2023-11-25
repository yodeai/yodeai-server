
import time
import random
import sys
import os
from openai import ChatCompletion
import re
import requests
import json
import openai
from DB import supabaseClient
import boto3

import google.generativeai as palm
palm.configure(api_key=os.getenv("GOOGLE_API_KEY"))


from dotenv import load_dotenv
load_dotenv(dotenv_path='.env.local')
openai.api_key = os.getenv("OPENAI_API_KEY")

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')


s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

import sys
import time
import random
import timeout_decorator
from timeout_decorator import timeout

def remove_leadingntrailing_special_chars(input_string):    
    cleaned_string = re.sub(r'^[\'"`\s]+', '', input_string)
    cleaned_string = re.sub(r'[\'"`]+$', '', cleaned_string)
    return cleaned_string

def exponential_backoff(retries=5, backoff_in_seconds=1, out=sys.stdout, timeout_in_seconds=None):
    if timeout_in_seconds is None:
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
    else:
        def backoff(func):
            def wrapper(*args, **kwargs):
                x = 0
                while True:
                    try:
                        result = timeout(timeout_in_seconds)(func)(*args, **kwargs)
                        return result
                    except Exception as e:
                        if x == retries:
                            out.write(f"Exception raised; number of retries is over {retries}\n")
                            raise e  # Re-raise the exception
                        if isinstance(e, timeout_decorator.timeout_decorator.TimeoutError):
                            out.write(f"Function timed out after {timeout_in_seconds} seconds; retrying\n")
                        else:
                            out.write(f"Exception raised; sleeping for a backoff of: {backoff_in_seconds * 2**x} seconds\n")
                        x += 1

                        sleep_duration = (backoff_in_seconds * 2**x + random.uniform(0, 1))
                        time.sleep(sleep_duration)
            return wrapper
    return backoff



@exponential_backoff(retries=6, backoff_in_seconds=1, out=sys.stdout)
def get_completion(prompt, model='models/text-bison-001'):    
    if (model == "gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]
        response = ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message["content"]   
    # the model is text-bison     
    completion = palm.generate_text(model='models/text-bison-001', prompt=prompt, temperature=0.2)
    cleaned_result = remove_leadingntrailing_special_chars(completion.result)
    return cleaned_result

## Below is OPENAI's get completition
# def get_completion(prompt, model="gpt-3.5-turbo"):
#     messages = [{"role": "user", "content": prompt}]
#     response = ChatCompletion.create(
#         model=model,
#         messages=messages,
#         temperature=0,
#     )
#     return response.choices[0].message["content"]



def fetchLinksFromDatabase():
    data, count = supabaseClient.table('links').select('title, url').execute()
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

######### Afshin: I commented out this code. It does not compile!
# def getDocumentsVectorStore():
#     client = supabaseClient
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     return SupabaseVectorStore(
#         client=client,
#         embedding=embeddings,
#         table_name="documents_huggingface",
#         query_name="match_documents_huggingface"
#     )
# def getQuestionsVectorStore():
#     client = supabaseClient
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     return SupabaseVectorStore(
#         client=client,
#         embedding=embeddings,
#         table_name="questions_huggingface",
#         query_name="match_questions_huggingface"
#     )


def remove_invalid_surrogates(text):
    # Remove invalid high surrogates
    text = re.sub(r'[\uD800-\uDBFF](?![\uDC00-\uDFFF])', '', text)
    # Remove invalid low surrogates
    text = re.sub(r'(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]', '', text)
    # Remove Unicode/null characters
    text = re.sub(r'\u0000', '', text)
    text = re.sub(r'\x00', '', text)
    return text

# Current options for model are "BGELARGE_MODEL" and "MINILM_MODEL"
@exponential_backoff(retries=6, backoff_in_seconds=1, out=sys.stdout)
def getEmbeddings(texts, model='BGELARGE_MODEL'):
    headers = {
        "Authorization": f"Bearer {os.environ.get('HUGGINGFACEHUB_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    # commenting Dana's code below 
    # if isinstance(texts, str):
    #     texts = [texts]
        
    data = {"inputs": texts}

    response = requests.post(os.environ.get(model), headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        print("Error in Hugging Face API Response in embeddings:", response.content.decode("utf-8"))
        return None

    try:
        response_content = response.content.decode("utf-8")
        embeddings = response.json()
    except json.JSONDecodeError:
        print("Error decoding JSON in Hugging Face API Response.")
        return None

    if embeddings is None or 'embeddings' not in embeddings:
        print("Error in getting embeddings.")
        return None

    return embeddings['embeddings']



def backportRows():
    data, count = supabaseClient.table('lens').select('owner_id, lens_id, updated_at').execute()
    data = data[1]
    for obj in data:
        obj["access_type"] = 'owner'
        obj["user_id"] = obj["owner_id"]
        del obj["owner_id"]
    supabaseClient.table('lens_users').upsert(data, ignore_duplicates=True, on_conflict='user_id, lens_id').execute()


def test_utils():    
    prompt = "I am using palm.generate_text(model='models/text-bison-001',prompt) to generate text. what other choices do i have besides text-bison-001? can you list the number of parameters of each, and that which one is more suitable for general purpose use in my code? specifically, can you compare them in terms of accuracy?"    
    response = get_completion(prompt, "gpt-3.5-turbo")
    print(response)

if __name__ == "__main__":  
    start_time = time.time()
    test_utils()
    print(f"time: {time.time()-start_time}")