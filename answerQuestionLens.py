
from utils import get_completion, getEmbeddings
from DB import supabaseClient
import time
from debug.tools import clearConsole
MODEL_NAME = "gpt-3.5-turbo"

relevanceThreshold = 5
notFound = "The question does not seem to be relevant to the provided content."
irrelevantText = 'Sorry, the question is irrelevant to the text.'
def answer_question_lens(question: str, lensID: str, activeComponent: str, userID: str, published: bool=False, google_user_id: str=None):
    #sys.stdout.write(lensID+" "+activeComponent+" "+userID)
    #clearConsole(" before embedding")
    start_time = time.time()
    # Record the start time for getRelDocs
    get_rel_docs_start_time = time.time()
    question_embedding=getEmbeddings(question)


    def getRelDocs(q,  match_count = 5):
        if published:
            rpc_params = {
            "lensid": lensID,
            "match_count": 5, 
            "query_embedding": question_embedding,
            }
            #sys.stdout.write("before DB call:\n")
            data, error = supabaseClient.rpc("get_top_chunks_for_lens_published", rpc_params).execute() 
            return data[1]

        if (activeComponent == "global" or activeComponent == "myblocks"):
            rpc_params = {
                "match_count": match_count, 
                "query_embedding": question_embedding,
                "user_id": userID ,
                "googleid": google_user_id,
            }
            print("len:")
            print(len(question_embedding))
            #sys.stdout.write("before DB call:\n")
            data, error = supabaseClient.rpc("get_top_chunks_google", rpc_params).execute() 
            return data[1]
        
        if (activeComponent == "inbox"):                        
            rpc_params = {
                "match_count": match_count, 
                "query_embedding": question_embedding,
                "googleid": google_user_id,
                "userid": userID,
            }
            data, error = supabaseClient.rpc("get_top_chunks_for_inbox_google", rpc_params).execute() 
            return data[1]
        #clearConsole(" calling lens func")
        
        
        rpc_params = {
            "lensid": lensID,
            "match_count": match_count, 
            "query_embedding": question_embedding,
            "googleid": google_user_id,
            "user_id": userID
        }
        
        # rpc_params = {
        #     "lensid": lensID,
        #     "match_count": match_count, 
        #     "query_embedding": question_embedding,
        # }        
        print("rpc_params", activeComponent)
        # print(userID)
        # print(google_user_id)
        # print(match_count)
        # print(question_embedding)
        #data, error = supabaseClient.rpc("get_top_chunks_for_lens", rpc_params).execute()               
        data, error = supabaseClient.rpc("get_top_chunks_for_lens_google", rpc_params).execute()               
        print("data,error:\n")
        print(data)
        print("\n\n")
        print(error)
        print("\n\n")
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
    # 1065
    print(f"Time taken by getRelDocs: {time.time() - get_rel_docs_start_time:.2f} seconds")

    relevant_block_ids = [d['block_id'] for d in relevant_chunks]
    print(relevant_block_ids)
    text = ""    
    for d in relevant_chunks:        
        text += d['content'] + "\n\n"        
    prompt = f"You are answering questions asked by a user. Answer the question: " + question + " in a helpful and concise way and in at most one paragraph, using the following text inside triple quotes:\n '''" + text + "''' \n >>"
    
    
    # Record the start time for get_completion
    get_completion_start_time = time.time()
    response = get_completion(prompt, MODEL_NAME)
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
        "user_id": userID,
        "block_ids": metadata["blocks"] if "blocks" in metadata else []
    }
    if lensID:
        # Check if data already exists
        existingData, count = supabaseClient.from_('questions') \
            .select('id') \
            .eq('question_text', insertData['question_text']) \
            .eq('lens_id', insertData['lens_id']) \
            .execute()
        
        if existingData is not None and len(existingData[1]) != 0:
            # A matching record already exists, get the question ID
            print("existing data", existingData)
            questionId = existingData[1][0]["id"]
            print(f'Data already exists. Question ID: {questionId}')
        else:
            # No matching record found, proceed with the insertion
            data, count = supabaseClient.table('questions') \
                .insert([insertData]) \
                .execute()
            questionId = data[1][0]["id"]
            print(f'Data inserted successfully. New Question ID: {questionId}')
    else:
        questionId = -1
    print("fully done!")
    return {
        "question": question,
        "answer": response,
        "metadata": metadata,  
        "question_id": questionId
    }


def test_answer_question_lens():
    #question = "what are some ways to develop autonomous language agents?"
    #lensID = "159"
    question = "what do birthdays look like? and what types for games do people play?"
    lensID = "190"
    # question = "What is the meaning of life?"
    # lensID = "6"
    user_id = "e6666aec-85eb-4873-a059-c7b2414f1b26"
    response = answer_question_lens(question, lensID, "global", user_id)    
    print(response)



def update_question_popularity(lensID, userID, questionID, diff):
    # First, perform a SELECT query to check if a row with the given data already exists
    existing_data, count = supabaseClient.table('question_votes') \
        .select()\
        .eq('user_id', userID)\
        .eq('question_id', questionID)\
        .eq('lens_id', lensID)\
        .execute()
    inserted = False
    # If existing_data is not None, a row with the given data already exists
    if len(existing_data[1]) != 0:
        # Perform an UPDATE operation
        data, count = supabaseClient.table('question_votes') \
            .update({'vote': diff})\
            .eq('user_id', userID)\
            .eq('question_id', questionID)\
            .eq('lens_id', lensID)\
            .execute()
    else:
        # Perform an INSERT operation
        inserted = True
        data, count = supabaseClient.table('question_votes') \
            .insert([{
                'user_id': userID,
                'question_id': questionID,
                'lens_id': lensID,
                'vote': diff
            }])\
            .execute()
        
    # update the vote count
    data, error = supabaseClient.rpc('increment', { "x": diff, "row_id": questionID, "lens_id": lensID }).execute()

    return {"data": data, "inserted": inserted, "error": error}

def get_searchable_feed(question, lensID):
    get_rel_docs_start_time = time.time()
    # Record the start time for getRelDocs
    get_rel_docs_start_time = time.time()
    def getRelDocs(q, match_count = 3):
        question_embedding=getEmbeddings(question, 'MINILM_MODEL')

        rpc_params = {
            "match_count": match_count, 
            "query_embedding": question_embedding,
            "lens_id": lensID
        }
        data, error = supabaseClient.rpc("match_questions_lens", rpc_params).execute() 
        return data[1]

    ans1 = getRelDocs(question)
    print(f"Time taken by getRelDocs: {time.time() - get_rel_docs_start_time:.2f} seconds")
    docs = []
    docs.extend(ans1)
    return {"questions": docs}

if __name__ == "__main__":  
    test_answer_question_lens(914)
    #get_searchable_feed("What time are the lectures of Integrative Biology 35ac?", 188)


    
