from DB import supabaseClient
from utils import getEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from urllib.parse import urlparse
import tempfile
from utils import s3
from utils import remove_invalid_surrogates
from utils import get_completion
import re

def replace_two_whitespace(input_string):
    result_string = re.sub(r'(\s)\1+', r'\1', input_string)
    return result_string

def extract_title(text):
    if (len(text) == 0):
        return "Untitled"
    if (len(text) > 1000):
        text = text[0:1000]    
    #prompt = f"Choose a title for the following document inside tripple quotes. What is the title? You response should contain the title alone. If there is no title, then respond with ``Untitiled''. Document: ```{text}'''"
    prompt = f"Choose a title for the following document inside tripple quotes. You response should contain the title alone. If you cannot find a natural title, then respond with ``Untitiled''. Document: ```{text}'''"
    response = get_completion(prompt)
    return response


def update_title(title, block_id):
    update_response, update_error = supabaseClient.table('block')\
        .update({'title': title})\
        .eq('block_id', block_id)\
        .execute()
    if not update_response or len(update_response) < 2 or not update_response[1]:
        raise Exception(f"Error updating title for block with id {block_id}: {update_error}")


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
    
if __name__ == "__main__":
    try:
        processBlock(363)
        print("Content processed and chunks stored successfully.")
    except Exception as e:
        print(f"Exception occurred: {e}")
