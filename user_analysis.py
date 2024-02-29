from DB import supabaseClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import remove_invalid_surrogates, get_completion
import time
from competitive_analysis import update_whiteboard_status
import re
from utils import getEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

MODEL_NAME = "gpt-3.5-turbo"
eps = 0.10

def get_block_ids(lens_id):
    block_ids, count = supabaseClient.table('lens_blocks').select("block_id").eq('lens_id', lens_id).execute()
    return block_ids[1]

def get_block_names(block_ids):
    # block_ids_list = [block_info['block_id'] for block_info in block_ids]
    block_ids_list = block_ids
    block_names, count = supabaseClient.table('block').select('block_id', 'title').in_('block_id', block_ids_list).execute()
    return block_names[1]

def get_block_content(block_id):
    block_content, count = supabaseClient.table('block').select('title', 'content').eq('block_id', block_id).execute()
    return block_content[1]

def split_text_into_chunks(document_content):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=0,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    chunks = text_splitter.split_text(document_content)
    return [remove_invalid_surrogates(text) for text in chunks]

def extract_background_info(block_id, background_info_embedding):
    rpc_params = {
    "interview_block_id": block_id,
    "matchcount": 5, 
    "queryembedding": background_info_embedding
    }
    
    data, error = supabaseClient.rpc("get_top_chunks_for_user_analysis", rpc_params).execute()
    relevant_chunks = data[1]
    text = ""
    for d in relevant_chunks:        
        text += d['content'] + "\n\n"  
    prompt = f"Please output a one sentence summary about the interviewee:  ```{text}'''."
    return get_completion(prompt, MODEL_NAME)

def get_topic_embedding(topics):
    result = {}
    for area in topics:
        embedding=getEmbeddings(area)
        result[area] = embedding
    return result

def cosine_similarity_vectors(vec1, vec2):
    # Ensure the vectors are not empty
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    # Calculate cosine similarity
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    return similarity

def find_closest_comment(comment_embedding, comment_embeddings):
    similarities = [cosine_similarity_vectors(comment_embedding, emb) for emb in comment_embeddings]
    max_similarity = max(similarities, default=0.0)
    return max_similarity

def generate_from_existing_topics(topics, lens_id, whiteboard_id, block_ids, new_percentage):
    topics_embedding = get_topic_embedding(topics)
    json_object = {"summary": {"users": [], "topics": [{"key": name, "name": name} for i, name in enumerate(topics)]},
                "insights": []}

    # block_ids  = get_block_ids(lens_id)
    block_infos = get_block_names(block_ids)
    background_info_embedding = getEmbeddings("background information about interviewee")

    for user_id, block_info in enumerate(block_infos):
        print("user", user_id)
        block_id = block_info["block_id"]
        name = block_info["title"]
        comment_summary = []
        block_content = get_block_content(block_id)

        background_info = extract_background_info(block_id, background_info_embedding)
        current_insights = {"data": [], "user": {"id": user_id, "info": background_info, "name": name}}
        comment_embeddings = []
        for topic_id, topic in enumerate(topics):
            rpc_params = {
            "interview_block_id": block_id,
            "matchcount": 5, 
            "queryembedding": topics_embedding[topic],
            }
            
            data, error = supabaseClient.rpc("get_top_chunks_for_user_analysis", rpc_params).execute()
            relevant_chunks = data[1]
            text = ""
            for d in relevant_chunks:        
                text += d['content'] + "\n\n"  

            prompt = f"Please provide a concise summary for the given content, if it is relevant to {topic}. If the content is irrelevant, just output a single bullet point: '- not relevant'.  If referring to the interviewee, refer to them as 'The user'. Start each bullet point with '-':\n\n```{text}```\n\nEnsure that the bullet points are relevant to {topic}, IF A BULLET POINT IS IRRELEVANT TO THE {topic} THEN DO NOT INCLUDE IT AT ALL. DO NOT OUTPUT MORE THAN 10 BULLET POINTS, AIM TO GENERATE LESS BULLET POINTS."
            bullet_summary = get_completion(prompt, MODEL_NAME)

            bullets = bullet_summary.split("- ")

            comments = {"comments": [{"id": i, "comment": bullet} for i, bullet in enumerate(bullets) if bullet != "" and bullet != "-" and bullet != " " and find_closest_comment(getEmbeddings(bullet), comment_embeddings) < 1-eps],
                        "topicKey": topic, "topicName": topic}
            filtered_comments = '. '.join([comment["comment"] for comment in comments["comments"]])
            if not filtered_comments:
                comments = {"comments": [{"id": 0, "comment": "not relevant"}],
                        "topicKey": topic, "topicName": topic}
            current_insights["data"].append(comments)
            if len(block_content[0]["content"]) < 3000:
                for comment in bullets:
                    comment_embeddings.append(getEmbeddings(comment))
            prompt = f"Please output a summary of LESS THAN 30 WORDS of this content:  ```{filtered_comments}'''. IF {filtered_comments} just says 'not relevant' or is blank then output 'not relevant'."
            summary = get_completion(prompt, MODEL_NAME)
            comment_summary.append({"content": summary, "topicKey": topic})
            data, error = supabaseClient.rpc("update_plugin_progress", {"id": whiteboard_id, "new_progress": new_percentage}).execute() 

        json_object["insights"].append(current_insights)
        json_object["summary"]["users"].append({"id": user_id, "name": name, "commentSummary": comment_summary})
    return json_object

def clean_insight_area(value):
    # Remove trailing whitespaces on the outside of phrases
    cleaned_value = re.sub(r'^\s+|\s+$', '', value)
    # Remove trailing punctuation
    cleaned_value = re.sub(r'[^\w\s]', '', cleaned_value)
    return cleaned_value


def splitTopics(input_string):
    # Splitting the string using "\n" and "."
    result = re.split(r'[\n.]', input_string)

    # Removing empty strings from the result
    result = [part.strip() for part in result if (len(part.strip())>3)]

    splits=[]
    for i, part in enumerate(result, 1):
        splits.append(part.lower().replace('*', ''))
    return splits

def generate_topics(lens_id, block_ids, new_percentage, whiteboard_id):
    # block_ids  = get_block_ids(lens_id)
    topics_text = ""
    for block_info in block_ids:
        block_id = block_info
        # call rpc function get_most_relevant_chunk to get top 5 chunks of each user that is most relevant to the user's avg_embedding
        rpc_params = {
        "interview_block_id": block_id,
        "matchcount": 5,
        }
        
        data, error = supabaseClient.rpc("get_most_relevant_chunk", rpc_params).execute()
        relevant_chunks = data[1]
        text = ""
        for d in relevant_chunks:        
            text += d['content'] + "\n\n"  
        if topics_text:
            prompt = f"Please update these 3 interview topics: {topics_text} according to this content: ```{text}''' and output the three topics again and START EACH BULLET WITH A '-' AND  MAKE THEM GENERALIZABLE AND 3-4 WORDS ONLY:."
            topics_text = get_completion(prompt, MODEL_NAME)
        else:
            prompt = f"Please output 3 main topics that can be extracted from this interview content, and START EACH BULLET WITH A '-' AND MAKE THEM GENERALIZABLE AND 3-4 WORDS ONLY :  ```{text}'''."
            topics_text = get_completion(prompt, MODEL_NAME)
        print("topics", topics_text)
    data, error = supabaseClient.rpc("update_plugin_progress", {"id": whiteboard_id, "new_progress": new_percentage}).execute() 
    result = splitTopics(topics_text)
    return result

    # generate 3 topics from the chunks
    # iteratively update the 3 topics with each user




def generate_user_analysis(topics, lens_id, whiteboard_id, block_ids=[]):
    try:
        start_time = time.time()
        print("topics", topics)
        update_whiteboard_status("processing", whiteboard_id)
        topics = [clean_insight_area(topic) for topic in topics]
        topics = [topic for topic in topics if topic != ""]
        if not block_ids:
            block_ids = get_block_ids(lens_id)
            block_ids = [block_info['block_id'] for block_info in block_ids]
        if not topics:
            num_cells = 3*len(block_ids)
            new_percentage = float(1/(num_cells + 1))
            topics = generate_topics(lens_id, block_ids, new_percentage, whiteboard_id)
            topics = [clean_insight_area(topic) for topic in topics]
            topics = [topic for topic in topics if topic != ""]
        else:
            num_cells = len(topics) * len(block_ids)
            new_percentage = float(1/(num_cells))
        
        json_object = generate_from_existing_topics(topics, lens_id, whiteboard_id, block_ids, new_percentage)

        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        return json_object
    except Exception as e:
        print(f"Error in task: {e}")
        update_whiteboard_status("error", whiteboard_id)
        raise


# topics = ["existing solutions and problems", "yodeai impressions and wants"]
# lens_id = 874
# print(generate_user_analysis(topics, lens_id))