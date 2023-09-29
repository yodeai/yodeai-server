
from utils import get_completion, getEmbeddings
from DB import mySupabase
import time

relevanceThreshold = 5
notFound = "The question does not seem to be relevant to the provided content."

def answer_question_lens(question: str, lensID: str):
    start_time = time.time()
    response = "This is a test response from the backend, and the question is: " + question + " and the lensID is: " + lensID
    # Record the start time for getRelDocs
    get_rel_docs_start_time = time.time()
    def getRelDocs(q):
        docs = []
        question_embedding=getEmbeddings(question)
 
        rpc_params = {
            "lensid": lensID,
            "match_count": 4, 
            "query_embedding": question_embedding,
        }
        data, error = mySupabase.rpc("get_top_chunks_for_lens", rpc_params).execute() 

        
        # data = mySupabase.from_('lens_blocks').select('block_id').eq('lens_id', lensID).execute().data    
        # block_ids = [d['block_id'] for d in data]   
        # relevant_chunks = mySupabase.from_('chunk').select('chunk_id').in_('block_id', block_ids).execute().data
        # chunk_ids = [d['chunk_id'] for d in relevant_chunks]
        # filter = {
        #     "chunk_id": {"in": chunk_ids}
        # }
        # question_embedding=getEmbeddings(question)
        # rpc_params = {
        #     "filter": filter,
        #     "match_count": 4,
        #     "query_embedding": question_embedding
        # }        
        # data, error =  mySupabase.rpc("match_chunks", rpc_params).execute()

        return data[1]
        
        #filter_params = {"chunk_id": {"$in_": chunk_ids}}
        #ans2 = vectorStore.similarity_search(question, 8, filter=filter)

    print("starting to get docs")
    
    relevant_chunks = getRelDocs(question)  
    print(f"Time taken by getRelDocs: {time.time() - get_rel_docs_start_time:.2f} seconds")

    #print("done with get docs: ", relevant_chunks)  
    relevant_block_ids = [d['block_id'] for d in relevant_chunks]
    text = ""
    #print("in results\n\n\n:")    
    for d in relevant_chunks:        
        #print(d)        
        text += d['content'] + "\n\n"        
    prompt = "You are answering questions asked by a user. Answer the question: " + question + " in a helpful and concise way and in at most one paragraph, using the following text inside tripple quotes: '''" + text + "''' \n <<<REMEMBER:  If the question is irrelevant to the text, do not try to make up an answer, just say that the question is irrelevant to the context.>>>"
    print("starting to get completion")
    # Record the start time for get_completion
    get_completion_start_time = time.time()
    response = get_completion(prompt)
    # Print the time taken by get_completion
    print(f"Time taken by get_completion: {time.time() - get_completion_start_time:.2f} seconds")
        
    #getRel = getRelevance(question, response, text)
    #if getRel == None or getRel <= relevanceThreshold:
    #    response = notFound
    metadata = {"blocks": list(set(relevant_block_ids))}
    print("done with process vector search")
    # Print the total time taken
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

    return {
        "question": question,
        "answer": response,
        "metadata": metadata  
    }


def test_answer_question_lens():
    #question = "what are some ways to develop autonomous language agents?"
    #lensID = "159"
    question = "how do spiders know that there's another spider on their net?"
    lensID = "1"
    # question = "What is the meaning of life?"
    # lensID = "6"
    response = answer_question_lens(question, lensID)
    print(response)
    

if __name__ == "__main__":
    test_answer_question_lens()
