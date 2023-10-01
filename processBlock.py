from DB import supabaseClient
from utils import getEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def processBlock(block_id):
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
     
    data, error = supabaseClient.table('block').select('content').eq('block_id', block_id).execute()
    
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
        supabaseClient.table('chunk').insert({
            'block_id': block_id,
            'content': chunk,
            'metadata': {},  
            'embedding': embedding,  
            'chunk_type': 'split',  
            'chunk_start': 0,
            'chunk_length': len(chunk)
        }).execute()
   
    # After processing all chunks, update the status of the block to 'ready'
    update_response, update_error = supabaseClient.table('block')\
        .update({'status': 'ready'})\
        .eq('block_id', block_id)\
        .execute()
    # Handle potential errors during the update
    if not update_response or len(update_response) < 2 or not update_response[1]:
        raise Exception(f"Error updating status for block with id {block_id}: {update_error}")
        

if __name__ == "__main__":
    try:
        processBlock(60)
        print("Content processed and chunks stored successfully.")
    except Exception as e:
        print(f"Exception occurred: {e}")
