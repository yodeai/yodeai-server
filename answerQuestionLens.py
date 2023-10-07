
from utils import get_completion, getEmbeddings
from DB import supabaseClient
import time
from debug.tools import clearConsole
import sys

relevanceThreshold = 5
notFound = "The question does not seem to be relevant to the provided content."

def answer_question_lens(question: str, lensID: str, userID: str):
    start_time = time.time()
    response = "This is a test response from the backend, and the question is: " + question + " and the lensID is: " + lensID
    # Record the start time for getRelDocs
    get_rel_docs_start_time = time.time()
    def getRelDocs(q):
        docs = []
        question_embedding=getEmbeddings(question)
        if (lensID == "NONE"):                        
            rpc_params = {
                "match_count": 5, 
                "query_embedding": question_embedding,
                "user_id": userID 
            }
            sys.stdout.write("before DB call:\n")
            data, error = supabaseClient.rpc("get_top_chunks", rpc_params).execute() 
            sys.stdout.write("after DB call:\n")
            sys.stdout.write(userID+"\n")
            sys.stdout.write(data[0])
            sys.stdout.write("\n\n-------------\n\n")
            sys.stdout.write(str(data[1]))
            sys.stdout.write("\n\n-------------\n\n")
            sys.stdout.write(error.__str__)
            return data[1]
               
        rpc_params = {
            "lensid": lensID,
            "match_count": 5, 
            "query_embedding": question_embedding,
        }
        data, error = supabaseClient.rpc("get_top_chunks_for_lens", rpc_params).execute() 
        return data[1]
        
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


        
        #filter_params = {"chunk_id": {"$in_": chunk_ids}}
        #ans2 = vectorStore.similarity_search(question, 8, filter=filter)

    print("starting to get docs")
    
    relevant_chunks = getRelDocs(question)  
    print(f"Time taken by getRelDocs: {time.time() - get_rel_docs_start_time:.2f} seconds")

    #print("done with get docs: ", relevant_chunks)  
    relevant_block_ids = [d['block_id'] for d in relevant_chunks]
    text = ""
    clearConsole ("results:") 
    for d in relevant_chunks:        
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
    lensID = "2"
    # question = "What is the meaning of life?"
    # lensID = "6"
    response = answer_question_lens(question, lensID)
    print(response)
    

if __name__ == "__main__":
            question_embedding=getEmbeddings("where can i find ice cream?")
            rpc_params = {
                "match_count": 5, 
                "query_embedding": question_embedding,
                "user_id": "e6666aec-85eb-4873-a059-c7b2414f1b26" 
            }
            clearConsole("before DB call:\n")
            data, error = supabaseClient.rpc("get_top_chunks", rpc_params).execute() 
            clearConsole("after DB call:")
            print(data)
    #test_answer_question_lens()
