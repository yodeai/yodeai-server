from DB import mySupabase
from utils import getEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import datetime

def processBlock(block_id):

    # Deleting existing chunks for the block_id before processing
    res =mySupabase.table('chunk').delete().eq('block_id', block_id).execute()
    if not res.data and res.count is not None:
        raise Exception(f"Error deleting chunks with block_id {block_id}: {res.error_msg if hasattr(res, 'error_msg') else 'Unknown error'}")
     
    data, error = mySupabase.table('block').select('content').eq('block_id', block_id).execute()
    
    if error and error[1]:
        raise Exception(f"Error fetching block with id {block_id}: {error[1]}")
    if not data or not data[1]:
        raise Exception(f"No block found with id {block_id}")
    if len(data[1]) > 1:
        raise Exception(f"Multiple blocks found with id {block_id}")
    
    content = data[1][0]['content']
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )

    # Splitting the content
    chunks = text_splitter.split_text(content)
    # Getting the embeddings for each chunk
    embeddings = getEmbeddings(chunks)
    
    if embeddings is None:
        raise Exception("Error in getting embeddings.")

    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        print (f"Processing chunk {idx} of block {block_id}")
        
        # Creating a new row in chunks table for each split
        mySupabase.table('chunk').insert({
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
        processBlock(36)
        print("Content processed and chunks stored successfully.")
    except Exception as e:
        print(f"Exception occurred: {e}")
