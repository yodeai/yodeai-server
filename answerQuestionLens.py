
from utils import get_completion, getEmbeddings
from DB import mySupabase

relevanceThreshold = 5
notFound = "The question does not seem to be relevant to the provided content."

def answer_question_lens(question: str, lensID: str):
    response = "This is a test response from the backend, and the question is: " + question + " and the lensID is: " + lensID

    def getRelDocs(q):
        docs = []
        #chunkIDList = mySupabase.from_('lens_chunks').select('*').eq('lens_id', lensID).execute()
        data = mySupabase.from_('lens_blocks').select('block_id').eq('lens_id', lensID).execute().data
        block_ids = [d['block_id'] for d in data]   
        relevant_chunks = mySupabase.from_('chunk').select('chunk_id').in_('block_id', block_ids).execute().data
        chunk_ids = [d['chunk_id'] for d in relevant_chunks]
        
        #print("\n\n\n"+chunk_ids.__str__())

        filter = {
            "chunk_id": {"in": chunk_ids}
        }
        question_embedding=getEmbeddings(question)
        rpc_params = {
            "filter": filter,
            "match_count": 4,
            "query_embedding": question_embedding
        }        
        data, error =  mySupabase.rpc("match_chunks", rpc_params).execute()       
        return data[1]
        
        #filter_params = {"chunk_id": {"$in_": chunk_ids}}
        #ans2 = vectorStore.similarity_search(question, 8, filter=filter)

    print("starting to get docs")
    relevant_chunks = getRelDocs(question)  
    #print("done with get docs: ", relevant_chunks)  
    relevant_block_ids = [d['block_id'] for d in relevant_chunks]
    text = ""
    #print("in results\n\n\n:")    
    for d in relevant_chunks:        
        #print(d)        
        text += d['content'] + "\n\n"        
    prompt = "You are answering questions asked by a user. Answer the question: " + question + " in a helpful and concise way and in at most one paragraph, using the following text inside tripple quotes: '''" + text + "''' \n <<<REMEMBER:  If the question is irrelevant to the text, do not try to make up an answer, just say that the question is irrelevant to the context.>>>"
    print("starting to get completion")
    response = get_completion(prompt);    
    #getRel = getRelevance(question, response, text)
    #if getRel == None or getRel <= relevanceThreshold:
    #    response = notFound
    metadata = {"blocks": list(set(relevant_block_ids))}
    print("done with process vector search")
    return {
        "question": question,
        "answer": response,
        "metadata": metadata  
    }


def test_answer_question_lens():
    question = "what are some ways to develop autonomous language agents?"
    lensID = "159"
    response = answer_question_lens(question, lensID)
    print(response)
    

if __name__ == "__main__":
    test_answer_question_lens()
