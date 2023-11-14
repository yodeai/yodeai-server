from DB import supabaseClient
from debug.tools import clearConsole
from utils import getEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from urllib.parse import urlparse
import tempfile
from utils import s3
from utils import remove_invalid_surrogates
from utils import get_completion
import re
import numpy as np
import openai
from langchain.chat_models import ChatOpenAI
from sklearn.cluster import KMeans
import math
from langchain.schema import Document



def replace_two_whitespace(input_string):
    result_string = re.sub(r'(\s)\1+', r'\1', input_string)
    return result_string

def extract_title(text):
    if (len(text) == 0):
        return "Untitled"
    if (len(text) > 1000):
        text = text[0:1000]    
    prompt = f"This is the beginning of a document. What's a good title for the document? Your response should include the title alone. \n Document: ```{text}'''"    
    response = get_completion(prompt, "gpt-3.5-turbo")
    return response


def get_preview(text):
    if (len(text) == 0):
        return "No content for a summary."
    if (len(text) > 3000):
        text = text[:3000]
    if (len(text) < 200):
        return text

    #prompt = f"You are generating a short summary for the following text inside triple qoutes in one or two sentences. This  summary will be shown to the user as a preview of  the entire text. It should be written as if it's part of the text; avoid language like ``this text studies''.  Text: ```{text}'''"
    prompt = f"You are generating a short summary for the following text inside triple qoutes in one or two sentences. This summary will be shown to the user as a preview of  the entire text. It should be written as if it's part of the text.  Text: ```{text}'''"
    response = get_completion(prompt, "gpt-3.5-turbo")
    return response

def stuff_summary(text): # very basic stuff summary, only work on smaller texts
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([text])
    stuff_prompt = """
    Write a concise summary of the following text delimited by triple backquotes.
    ```{text}```
    SUMMARY:
    """
    llm3 = ChatOpenAI(temperature=0,
                 openai_api_key="OPENAI_API_KEY",
                 max_tokens=1000,
                 model='gpt-3.5-turbo')
    stuff_prompt_template = PromptTemplate(template=stuff_prompt, input_variables=["text"])
    stuff_chain = load_summarize_chain(llm=llm3, chain_type="stuff", prompt=stuff_prompt_template)
    return stuff_chain.run(docs)

def brv_summary(text): # this should be used when the text is very long
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000)
    docs = text_splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(openai_api_key="OPENAI_API_KEY")
    vectors = embeddings.embed_documents([x.page_content for x in docs])
    num_clusters = int(math.sqrt(len(docs)))+1
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    closest_indices = []
    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)
    selected_indices = sorted(closest_indices)
    llm3 = ChatOpenAI(temperature=0,
                 openai_api_key="OPENAI_API_KEY",
                 max_tokens=1000,
                 model='gpt-3.5-turbo')
    map_prompt = """You will be given a part of a book/long article. This section will be enclosed in triple backticks (```)
    Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
    ```{text}```
    FULL SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    map_chain = load_summarize_chain(llm=llm3,
                             chain_type="stuff",
                             prompt=map_prompt_template)
    selected_docs = [docs[doc] for doc in selected_indices]
    summary_list = []
    for i, doc in enumerate(selected_docs):
        chunk_summary = map_chain.run([doc])
        summary_list.append(chunk_summary)
    summaries = "\n".join(summary_list)
    summaries = Document(page_content=summaries)
    llm4 = ChatOpenAI(temperature=0,
                 openai_api_key="OPENAI_API_KEY",
                 max_tokens=3000,
                 model='gpt-4')
    combine_prompt = """
    You will be given a series of summaries from a book/long article. The summaries will be enclosed in triple backticks (```)
    Your goal is to give a brief summary of what happened in the book.
    ```{text}```
    OVERALL SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    reduce_chain = load_summarize_chain(llm=llm4,
                             chain_type="stuff",
                             prompt=combine_prompt_template)
    return reduce_chain.run([summaries])



def map_reduce_summary(text): # map reduce summary, good for medium length text
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([text])
    llm = ChatOpenAI(temperature=0, openai_api_key="OPENAI_API_KEY")
    map_prompt = """Write a concise summary of the following: "{text}" CONCISE SUMMARY:"""
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    summary_chain = load_summarize_chain(llm=llm,
                                        chain_type='map_reduce',
                                        map_prompt=map_prompt_template)
    return summary_chain.run(docs)




def update_title(title, block_id):
    update_response, update_error = supabaseClient.table('block')\
        .update({'title': title})\
        .eq('block_id', block_id)\
        .execute()
    if not update_response or len(update_response) < 2 or not update_response[1]:
        raise Exception(f"Error updating title for block with id {block_id}: {update_error}")


def update_preview(preview, block_id):
    update_response, update_error = supabaseClient.table('block')\
        .update({'preview': preview})\
        .eq('block_id', block_id)\
        .execute()
    if not update_response or len(update_response) < 2 or not update_response[1]:
        raise Exception(f"Error updating preview for block with id {block_id}: {update_error}")


def processBlock(block_id):
    try:
        existing_row, error = supabaseClient.table('block') \
        .select('updated_at', 'created_at', 'owner_id') \
        .eq('block_id', block_id) \
        .execute()
    except Exception as e:
        print(f"Exception occurred while retrieving updated_at, created_at: {e}")

    # if the block_id does not exist in the block table, then add it to the inbox
    if existing_row[1][0]['created_at'] == existing_row[1][0]['updated_at']:                                
        try:                               
            insert_response, insert_error = supabaseClient.table('inbox') \
            .insert([{'user_id': existing_row[1][0]['owner_id'], 'block_id': block_id}]) \
            .execute()
            if insert_response and 'error' not in insert_response:
                print("Insertion to inbox successful.")
            else:
                print("Insertion to inbox failed. ")    
        except Exception as e:
            print(f"Exception occurred while adding block to inbox: {e}")

 
    # Update the status of the block to 'processing'
    update_response, update_error = supabaseClient.table('block')\
        .update({'status': 'processing'})\
        .eq('block_id', block_id)\
        .execute()

    # Handle potential errors during the update
    if not update_response or len(update_response) < 2 or not update_response[1]:
        raise Exception(f"Error updating status for block with id {block_id}: {update_error}")
    
    # Deleting existing chunks for the block_id before processing
    res =supabaseClient.table('chunk').delete().eq('block_id', block_id).execute()
    if not res.data and res.count is not None:
        raise Exception(f"Error deleting chunks with block_id {block_id}: {res.error_msg if hasattr(res, 'error_msg') else 'Unknown error'}")
     
    data, error = supabaseClient.table('block').select('content','block_type','file_url').eq('block_id', block_id).execute()
    
    if error and error[1]:
        raise Exception(f"Error fetching block with id {block_id}: {error[1]}")
    if not data or not data[1]:
        raise Exception(f"No block found with id {block_id}")
    if len(data[1]) > 1:
        raise Exception(f"Multiple blocks found with id {block_id}")
    
    #clearConsole(data)
    if (data[1][0]['block_type'] == "note"):
        content = data[1][0]['content']
    elif (data[1][0]['block_type'] == "pdf"):
        parsed_url = urlparse(data[1][0]['file_url'])
        bucket_name = parsed_url.netloc.split('.')[0]  # Extracts 'yodeai' from the hostname
        key_name = parsed_url.path.lstrip('/')  # Removes the leading '/'
        print ("bucket_name: "+ bucket_name+ " key_name: "+ key_name)
        # Fetch the PDF file from S3
        s3_object = s3.get_object(Bucket=bucket_name, Key=key_name)
        print ("s3_object: "+ str(s3_object))
        pdf_file = s3_object['Body']        
        #print ("pdf_file: "+ str(pdf_file))
        
        # Write the content to a temporary file
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_file.read())
            temp_pdf.flush()

            # Use your PyPDFLoader to load the PDF content
            loader = PyPDFLoader(temp_pdf.name)
            pages = []
            pages.extend(loader.load())
            content = ""
            for page in pages:
                content = content + page.page_content
        pdf_title = extract_title(content)
        update_title(pdf_title, block_id)
 
    content = replace_two_whitespace(content)
    content_preview = get_preview(content)
    update_preview(content_preview, block_id)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
        #["\n\n", "\n", "(?<=\. )", " ", ""]
    )

    # Splitting the content
    chunks = text_splitter.split_text(content)
    cleaned_chunks = [remove_invalid_surrogates(text) for text in chunks]
    # Getting the embeddings for each chunk
    embeddings = getEmbeddings(cleaned_chunks)
    
    if embeddings is None:
        raise Exception("Error in getting embeddings.")
    
    for idx, (chunk, embedding) in enumerate(zip(cleaned_chunks, embeddings)):
        
        #print (f"Processing chunk {idx} of block {block_id}")
        
        # Creating a new row in chunks table for each split
        supabaseClient.table('chunk').insert({
            'block_id': block_id,
            'content': chunk,
            'metadata': {},  
            'embedding': embedding,  
            'chunk_type': 'split',  
            'chunk_start': 0,
            'chunk_length': len(chunk)
        }).execute()
        

        #clearConsole(embeddings[0])
        np_embeddings = np.array(embeddings)        
        np_sum = np_embeddings.sum(axis=0)
        np_ave = np_sum/len(embeddings)
        ave_embedding = np_ave.tolist()
        
        #update_object = {'ave_embedding': ave_embeddings}
        response, error = supabaseClient.table('block').update({'ave_embedding': ave_embedding}).eq('block_id', block_id).execute()
        if error[1]:
            print("Error updating ave_embedding:", error)
        update_response, update_error = supabaseClient.table('block')\
        .update({'status': 'ready'})\
        .eq('block_id', block_id)\
        .execute()
    return content_preview
        


if __name__ == "__main__":
    try:
        #580,555
        blockIDs = []
        for b in blockIDs:
            processBlock(b)
        print("Content processed and chunks stored successfully.")
    except Exception as e:
        print(f"Exception occurred: {e}")
