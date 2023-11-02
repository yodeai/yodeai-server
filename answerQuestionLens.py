
from utils import get_completion, getEmbeddings
from DB import supabaseClient
import time
from debug.tools import clearConsole


relevanceThreshold = 5
notFound = "The question does not seem to be relevant to the provided content."
irrelevantText = 'Sorry, the question is irrelevant to the text.'
def answer_question_lens(question: str, lensID: str, activeComponent: str, userID: str):
    #sys.stdout.write(lensID+" "+activeComponent+" "+userID)
    #clearConsole(" before embedding")
    start_time = time.time()
    # Record the start time for getRelDocs
    get_rel_docs_start_time = time.time()
    question_embedding=getEmbeddings(question)      
    def getRelDocs(q):
        docs = []

        #clearConsole(" q embedding generated")  

        if (activeComponent == "global"):                        
            rpc_params = {
                "match_count": 5, 
                "query_embedding": question_embedding,
                "user_id": userID 
            }
            #sys.stdout.write("before DB call:\n")
            data, error = supabaseClient.rpc("get_top_chunks", rpc_params).execute() 
            return data[1]
        
        if (activeComponent == "inbox"):                        
            rpc_params = {
                "match_count": 5, 
                "query_embedding": question_embedding,
                "id": userID 
            }
            data, error = supabaseClient.rpc("get_top_chunks_from_inbox", rpc_params).execute() 
            return data[1]
        #clearConsole(" calling lens func")
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

    relevant_block_ids = [d['block_id'] for d in relevant_chunks]
    text = ""    
    for d in relevant_chunks:        
        text += d['content'] + "\n\n"        
    prompt = f"You are answering questions asked by a user. Answer the question: " + question + " in a helpful and concise way and in at most one paragraph, using the following text inside triple quotes:\n '''" + text + "''' \n <<<REMEMBER:  If you cannot find an answer with the given text in triple quotes, just return the following text:" + irrelevantText+ ">>>"
    clearConsole(prompt)
    
    
    # Record the start time for get_completion
    get_completion_start_time = time.time()
    response = get_completion(prompt)
    # Print the time taken by get_completion
    print(f"Time taken by get_completion: {time.time() - get_completion_start_time:.2f} seconds")
        
    #getRel = getRelevance(question, response, text)
    #if getRel == None or getRel <= relevanceThreshold:
    #    response = notFound
    if response == irrelevantText:
        metadata = {}
    else:
        metadata = {"blocks": list(set(relevant_block_ids))}
    
    print("done with process vector search")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")
    print("inserting data now")
    insertData = {
        "embedding": question_embedding,
        "popularity": 0,
        "question_text": question,
        "answer_full": response,
        "lens_id": lensID,
        "user_id": userID
    }
    try:
        data, count = supabaseClient.table('questions').insert(insertData).execute()
    except Exception as e:
        print("Error inserting into database", e)
    
    print("fully done!")
    return {
        "question": question,
        "answer": response,
        "metadata": metadata,  
    }


def test_answer_question_lens():
    #question = "what are some ways to develop autonomous language agents?"
    #lensID = "159"
    question = "what do birthdays look like?"
    lensID = "190"
    # question = "What is the meaning of life?"
    # lensID = "6"
    user_id = "e6666aec-85eb-4873-a059-c7b2414f1b26"
    response = answer_question_lens(question, lensID, "lens", userID )    
    print(response)



def update_question_popularity(id, diff, lensID):
    data, error = supabaseClient.rpc('increment', { "x": diff, "row_id": id, "lens_id": lensID }).execute()
    print("updated question popularity", data, error)
    return {"data": data, "error": error}

def get_searchable_feed(question, lensID):
    get_rel_docs_start_time = time.time()
    # Record the start time for getRelDocs
    get_rel_docs_start_time = time.time()
    def getRelDocs(q):
        question_embedding=getEmbeddings(question, 'MINILM_MODEL')
 
        rpc_params = {
            "match_count": 3, 
            "query_embedding": question_embedding,
            "lens_id": lensID
        }
        data, error = supabaseClient.rpc("match_questions_lens", rpc_params).execute() 
        return data[1]

    ans1 = getRelDocs(question)
    print(f"Time taken by getRelDocs: {time.time() - get_rel_docs_start_time:.2f} seconds")
    docs = []
    metadataList = []
    docs.extend(ans1)
    for doc in docs:
        metadataList.append(doc["metadata"])
    return {"documents": docs, "metadata": metadataList}

if __name__ == "__main__":
    # q = "what are fun birthdays like?"
    # userID = "e6666aec-85eb-4873-a059-c7b2414f1b26"
    # response = answer_question_lens(q, "NONE", userID)
    # print(response)
    question = "what are fun birthdays like?"
    userID = "e6666aec-85eb-4873-a059-c7b2414f1b26"
    lensID = "NONE"
    response = answer_question_lens(question, lensID, "inbox", userID )    
    print(response)
    test_answer_question_lens()
