
import time
import random
import sys
import os
from langchain import PromptTemplate
from openai import ChatCompletion
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

import sys
import time
import random
import timeout_decorator
from timeout_decorator import timeout

def exponential_backoff(retries=7, backoff_in_seconds=1, out=sys.stdout, timeout_in_seconds=40):
    def backoff(func):
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    result = timeout(timeout_in_seconds*(x+1))(func)(*args, **kwargs)
                    return result
                except Exception as e:
                    if x == retries:
                        out.write(f"Exception raised; number of retries is over {retries}\n")
                        raise e  # Re-raise the exception
                    if isinstance(e, timeout_decorator.timeout_decorator.TimeoutError):
                        print("got it")
                        out.write(f"Function timed out after {timeout_in_seconds} seconds; retrying\n")
                    else:
                        out.write(f"Exception raised; sleeping for a backoff of: {backoff_in_seconds * 2**x} seconds\n")
                    x += 1

                    sleep_duration = (backoff_in_seconds * 2**x + random.uniform(0, 1))
                    time.sleep(sleep_duration)
        return wrapper
    return backoff

# def exponential_backoff(retries=5, backoff_in_seconds=1, out=sys.stdout):
#     def backoff(func):
#         def wrapper(*args, **kwargs):
#             x = 0
#             while True:
#                 try:
#                     return func(*args, **kwargs)
#                 except:
#                     if x == retries:
#                         out.write(f"exception raised; number of retries is over {retries}\n")
#                         raise
#                     sleep_duration = (backoff_in_seconds * 2**x + random.uniform(0, 1))
#                     out.write(f"exception raised; sleeping for: {sleep_duration} seconds\n")
#                     time.sleep(sleep_duration)
#                     x += 1
#         return wrapper
#     return backoff


@exponential_backoff(retries=6, backoff_in_seconds=1, out=sys.stdout)
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

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
    text = re.sub(r'\\u0000', '', text)
    text = re.sub(r'\x00', '', text)
    return text

# Current options for model are "BGELARGE_MODEL" and "MINILM_MODEL"
@exponential_backoff(retries=6, backoff_in_seconds=1, out=sys.stdout)
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


def get_title(text):
    prompt =  f"Give the following text an informative title that reflects the main theme of the text: ```{text}''' Your answer should be formatted as follows: in the first line, write ``Title:'' followed with the title."    
    response = get_completion(prompt)
    title_starts = 1+response.find(":")
    title = response[title_starts:].strip()
    return title        


from langchain.text_splitter import RecursiveCharacterTextSplitter
def simply_summarize(text, cutoff=20000): # map reduce summary, good for medium length text
    print(f"starting simply summarize with length {len(text)}")        
    if (len(text)>cutoff):
        text = text[0:cutoff]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        separators=["\n\n", "\n", " ", ""]
        #["\n\n", "\n", "(?<=\. )", " ", ""]
    )

    chunks = text_splitter.split_text(text)
    cleaned_chunks = [remove_invalid_surrogates(text) for text in chunks]
    
    summary_list = []
    for index, chunk in enumerate(cleaned_chunks):
        print(f"starting to summarize chunk {index} with length {len(chunk)}")        
        prompt =  f"You are summarizing the following text given in triple qoutes. The summary should be concise, should try to preserve concrete information that are central to the narrative of the text, and should be at most 200 to 300 words. Text: ```{chunk}'''\n"
        summary_list.append(get_completion(prompt))

    chunk_list = ""
    for summary in summary_list:
        chunk_list += (summary+"\n") 
    prompt =  f"You are summarizing the following text given in triple qoutes.  The summary should be concise, should try to preserve concrete information that are central to the narrative of the text, and should be at most 200 to 300 words. Text: ```{chunk_list}'''\n"    
    return get_completion(prompt)


#from tiktoken import Tokenizer
def isOkay4openAILimit(text):
    return (len(text)<=20000) #because the tokenizer below doesn't work
    # tokenizer = Tokenizer()        
    # count = tokenizer.count_tokens(text)
    # return (count <= 4187) 




#################some summarization methods below
# def stuff_summary(text): # very basic stuff summary, only work on smaller texts
#     text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=500)
#     docs = text_splitter.create_documents([text])
#     stuff_prompt = """
#     Write a concise summary of the following text delimited by triple backquotes.
#     ```{text}```
#     SUMMARY:
#     """
#     llm3 = ChatOpenAI(temperature=0,
#                  openai_api_key="OPENAI_API_KEY",
#                  max_tokens=1000,
#                  model='gpt-3.5-turbo')
#     stuff_prompt_template = PromptTemplate(template=stuff_prompt, input_variables=["text"])
#     stuff_chain = load_summarize_chain(llm=llm3, chain_type="stuff", prompt=stuff_prompt_template)
#     return stuff_chain.run(docs)

# def brv_summary(text): # this should be used when the text is very long
#     text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000)
#     docs = text_splitter.create_documents([text])
#     embeddings = OpenAIEmbeddings(openai_api_key="OPENAI_API_KEY")
#     vectors = embeddings.embed_documents([x.page_content for x in docs])
#     num_clusters = int(math.sqrt(len(docs)))+1
#     kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
#     closest_indices = []
#     for i in range(num_clusters):
#         distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
#         closest_index = np.argmin(distances)
#         closest_indices.append(closest_index)
#     selected_indices = sorted(closest_indices)
#     llm3 = ChatOpenAI(temperature=0,
#                  openai_api_key="OPENAI_API_KEY",
#                  max_tokens=1000,
#                  model='gpt-3.5-turbo')
#     map_prompt = """You will be given a part of a book/long article. This section will be enclosed in triple backticks (```)
#     Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
#     ```{text}```
#     FULL SUMMARY:
#     """
#     map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
#     map_chain = load_summarize_chain(llm=llm3,
#                              chain_type="stuff",
#                              prompt=map_prompt_template)
#     selected_docs = [docs[doc] for doc in selected_indices]
#     summary_list = []
#     for i, doc in enumerate(selected_docs):
#         chunk_summary = map_chain.run([doc])
#         summary_list.append(chunk_summary)
#     summaries = "\n".join(summary_list)
#     summaries = Document(page_content=summaries)
#     llm4 = ChatOpenAI(temperature=0,
#                  openai_api_key="OPENAI_API_KEY",
#                  max_tokens=3000,
#                  model='gpt-4')
#     combine_prompt = """
#     You will be given a series of summaries from a book/long article. The summaries will be enclosed in triple backticks (```)
#     Your goal is to give a brief summary of what happened in the book.
#     ```{text}```
#     OVERALL SUMMARY:
#     """
#     combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
#     reduce_chain = load_summarize_chain(llm=llm4,
#                              chain_type="stuff",
#                              prompt=combine_prompt_template)
#     return reduce_chain.run([summaries])


# from langchain.chat_models import ChatOpenAI
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.summarize import load_summarize_chain

# def map_reduce_summary(text): # map reduce summary, good for medium length text
#     text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=500)
#     docs = text_splitter.create_documents([text])
#     llm = ChatOpenAI(temperature=0, openai_api_key=openai.api_key)
#     map_prompt = """Write a concise summary of the following: "{text}" CONCISE SUMMARY:"""
#     map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
#     summary_chain = load_summarize_chain(llm=llm,
#                                         chain_type='map_reduce',
#                                         map_prompt=map_prompt_template)
#     return summary_chain.run(docs)
