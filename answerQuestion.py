
from utils import addHyperlinksToResponse, fetchLinksFromDatabase, removeDuplicates, getRelevance, get_completion, getEmbeddings
from DB import supabaseClient
import time

relevanceThreshold = 5
notFound = "The question does not seem to be relevant to the provided content."


def answer_question(question, whatsappDetails=None):
    print("starting to embed question")
    question_embedding=getEmbeddings(question)
    result = process_vector_search(question)
    sources = []
    print("starting to sort metadata")
    print(result)
    for index, meta in enumerate(result["metadata"]):
      isTempURL = meta["source"] == "/var/folders/1r/n3tszc0n3zjcxyjf1tby4ng80000gn/T/tmpumhyx40m"
      title = "Campus Policies and Guidelines Concerning the Academic Calendar, RRR Week, Exams, and Commencement" if isTempURL else meta["title"]
      sourceURL = "https://registrar.berkeley.edu/wp-content/uploads/2021/03/050714_Campus-Policies-and-Guidelines-Concerning-the-Academic-Calendar.pdf" if isTempURL else meta["source"]
      sources.append(f"{index+1}. [{title}]({sourceURL})")
    sources = "\n".join(sources)
    linkMap = fetchLinksFromDatabase()
    if result["response"]:
        fullAnswer = addHyperlinksToResponse(result["response"], linkMap)
    else:
        fullAnswer = ""
    fullAnswer_with_sources = f"{fullAnswer}\n\n\nSources:\n{sources}"
    print("preparing insert data")
    insertData = {
        "embedding" : question_embedding,
        "popularity": 0,
        "question_text": result["question"],
        "answer_preview": result["response"],
        "answer_full": fullAnswer_with_sources,
        "asked_on_whatsapp": whatsappDetails != None
    }

    if (whatsappDetails):
        insertData['whatsapp_message_id'] = whatsappDetails.messageId
        insertData['whatsapp_phone_number'] = whatsappDetails.phoneNumber
    print("inserting data now")
    try:
        data, count = supabaseClient.table('questions').insert(insertData).execute()
    except:
        print("Error inserting into database")
    
    print("fully done!")
    return {
        "answer_preview": insertData["answer_preview"],
        "answer_full": insertData["answer_full"],
        "slug": "TODO"
    }

def process_vector_search(question: str) -> str:
    print("starting to answer question")
    get_rel_docs_start_time = time.time()
    # Record the start time for getRelDocs
    def getRelDocs(q):
        question_embedding=getEmbeddings(question)
 
        rpc_params = {
            "match_count": 4, 
            "query_embedding": question_embedding,
        }
        data, error = supabaseClient.rpc("match_documents", rpc_params).execute() 
        return data[1]

    print("starting to get docs")
    results = getRelDocs(question)
    print(f"Time taken by getRelDocs: {time.time() - get_rel_docs_start_time:.2f} seconds")
    dlist = results
    metadataList = [r["metadata"] for r in results]
    text = ""
    for d in dlist:
        text += d["content"] + "\n\n"
    prompt = "You are answering questions from freshmen at UC Berkeley. Answer the question: " + question + " in a helpful and concise way and in at most one paragraph, using the following text inside tripple quotes: '''" + text + "''' \n <<<REMEMBER:  If the question is irrelevant to the text, do not try to make up an answer, just say that the question is irrelevant to the context.>>>"
    print("starting to get completion")
    response = get_completion(prompt);    
    filteredMetadata = [meta for meta in metadataList if isinstance(meta["source"], str) and "title" in meta]
    uniqueMetadataList = removeDuplicates(filteredMetadata)
    getRel = getRelevance(question, response, text)
    if getRel == None or getRel <= relevanceThreshold:
        response = notFound
    print("done with process vector search")
    return {
        "question": question,
        "response": response,
        "metadata": uniqueMetadataList  
    }

def update_question_popularity(id, diff):
    data, error = supabaseClient.rpc('increment', { "x": diff, "row_id": id }).execute()
    print("updated question popularity", data, error)
    return {"data": data, "error": error}

def get_searchable_feed(question):
    get_rel_docs_start_time = time.time()
    # Record the start time for getRelDocs
    get_rel_docs_start_time = time.time()
    def getRelDocs(q):
        question_embedding=getEmbeddings(question, 'MINILM_MODEL')
 
        rpc_params = {
            "match_count": 3, 
            "query_embedding": question_embedding,
        }
        data, error = supabaseClient.rpc("match_questions", rpc_params).execute() 
        return data[1]

    ans1 = getRelDocs(question)
    print(f"Time taken by getRelDocs: {time.time() - get_rel_docs_start_time:.2f} seconds")
    docs = []
    metadataList = []
    docs.extend(ans1)
    for doc in docs:
        metadataList.append(doc["metadata"])
    return {"documents": docs, "metadata": metadataList}

