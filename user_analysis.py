from DB import supabaseClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import remove_invalid_surrogates, get_completion
import time
from competitive_analysis import update_whiteboard_status
import re
from utils import getEmbeddings
MODEL_NAME = "gpt-3.5-turbo"

def get_block_ids(lens_id):
    block_ids, count = supabaseClient.table('lens_blocks').select("block_id").eq('lens_id', lens_id).execute()
    return block_ids[1]

def get_block_names(block_ids):
    block_ids_list = [block_info['block_id'] for block_info in block_ids]
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

def extract_background_info(content):
    prompt = f"This content comes from a user interview, please return background about this interviewee in ONE SENTENCE:```{content}'''"
    return get_completion(prompt, MODEL_NAME)

def get_topic_embedding(topics):
    result = {}
    for area in topics:
        embedding=getEmbeddings(area)
        result[area] = embedding
    return result

def generate_from_existing_topics(topics, lens_id, whiteboard_id):
    update_whiteboard_status("processing", whiteboard_id)
    topics_embedding = get_topic_embedding(topics)
    json_object = {"summary": {"users": [], "topics": [{"key": name, "name": name} for i, name in enumerate(topics)]},
                "insights": []}

    block_ids  = get_block_ids(lens_id)
    block_names = get_block_names(block_ids)
    num_cells = len(topics) * len(block_names)

    for user_id, block_info in enumerate(block_names):
        print("user", user_id)
        block_id = block_info["block_id"]
        name = block_info["title"]
        comment_summary = []

        block_content = get_block_content(block_id)
        document_content = block_content[0]["content"]
        cleaned_chunks = split_text_into_chunks(document_content)
        background_info = extract_background_info(cleaned_chunks[0])
        current_insights = {"data": [], "user": {"id": user_id, "info": background_info, "name": name}}

        for topic_id, topic in enumerate(topics):
            rpc_params = {
            "interview_block_id": block_id,
            "matchcount": 5, 
            "queryembedding": topics_embedding[topic],
            }
            
            data, error = supabaseClient.rpc("get_top_chunks_for_user_analysis", rpc_params).execute()
            relevant_chunks = data[1]
            print("chunks", relevant_chunks)
            text = ""
            for d in relevant_chunks:        
                text += d['content'] + "\n\n"  

            prompt = f"Please output a max of 10 bullet points for this content, and start each bullet with '-':  ```{text}'''."
            bullet_summary = get_completion(prompt, MODEL_NAME)

            comments = {"comments": [{"id": i, "comment": bullet} for i, bullet in enumerate(bullet_summary.split("- ")) if bullet != ""],
                        "topicKey": topic, "topicName": topic}
            current_insights["data"].append(comments)

            prompt = f"Please output a summary of a MAXIMUM OF 30 WORDS for these bulletted chunks:  ```{bullet_summary}'''."
            summary = get_completion(prompt, MODEL_NAME)

            comment_summary.append({"content": summary, "topicKey": topic})
            new_percentage = float(1/(num_cells))
            data, error = supabaseClient.rpc("update_plugin_progress", {"id": whiteboard_id, "new_progress": new_percentage}).execute() 

        json_object["insights"].append(current_insights)
        json_object["summary"]["users"].append({"id": user_id, "name": name, "commentSummary": comment_summary})
    return json_object

def generate_from_scratch(lens_id, whiteboard_id):
    update_whiteboard_status("processing", whiteboard_id)
    json_object = {"summary": {"users": [], "topics": []}, "insights": []}

    block_ids = get_block_ids(lens_id)
    block_names = get_block_names(block_ids)
    num_cells = 4 * len(block_names)  # Generate 3 topics for each block

    for user_id, block_info in enumerate(block_names):
        print("user scratch", user_id)
        block_id = block_info["block_id"]
        name = block_info["title"]
        comment_summary = []

        block_content = get_block_content(block_id)
        document_content = block_content[0]["content"]
        cleaned_chunks = split_text_into_chunks(document_content)
        background_info = extract_background_info(cleaned_chunks[0])
        current_insights = {"data": [], "user": {"id": user_id, "info": background_info, "name": name}}

        # Initialize a list to store scored topics
        potential_topics = ""

        # Loop through all chunks and generate scored topics
        for chunk_id, chunk_content in enumerate(cleaned_chunks):
            if potential_topics:
                prompt = f"We have these 3 existing topics that are worded in phrases: {potential_topics}. Based on this content: ```{chunk_content}```, update the 3 existing topics if needed, and OUTPUT THEM IN A COMMA SEPARATED LINE. YOU SHOULD ONLY EITHER UPDATE OR CHANGE ONE OF THE 3 TOPICS, AND DO NOT OUTPUT MORE THAN 3 TOPICS. AGAIN, THEY SHOULD BE COMMA SEPARATED, NOT BULLETTED"
            else:
                prompt = f"Generate ONLY 3 topics related to: ```{chunk_content}```, and OUTPUT THEM IN A COMMA SEPARATED LINE, WITH EACH TOPIC CONSTRAINED TO A SHORT PHRASE. AGAIN, THEY SHOULD BE COMMA SEPARATED AND NOT BULLETTED, THERE SHOULD BE NO '-'"
            
            result = get_completion(prompt, MODEL_NAME)
            if result != "":
                potential_topics = result

            print("potential_topics", potential_topics)
        # Sort scored topics based on the score in descending order
        sorted_topics = potential_topics.split(",")
        sorted_topics = [topic.strip() for topic in sorted_topics]
        sorted_topics = sorted_topics[:3]
        print("sorted_topics", sorted_topics)
        new_percentage = float(1/(num_cells))
        data, error = supabaseClient.rpc("update_plugin_progress", {"id": whiteboard_id, "new_progress": new_percentage}).execute() 

        # Select the top 3 scored topics
        for topic in sorted_topics:
            print("topic", topic)
            bullet_summary = ""
            for chunk_id, chunk_content in enumerate(cleaned_chunks):
                prompt = f"Please output one bullet point summary of:  ```{chunk_content}''' that relates to {topic}, where each bullet point starts with a '-', AND PLEASE LIMIT TO 1 BULLET POINT."
                response = get_completion(prompt, MODEL_NAME)
                bullet_summary += response

            comments = {"comments": [{"id": i, "comment": bullet} for i, bullet in enumerate(bullet_summary.split("- ")) if bullet != ""],
                        "topicKey": topic, "topicName": topic}
            current_insights["data"].append(comments)

            prompt = f"Please output a summary of a MAXIMUM OF 30 WORDS for these bulletted chunks:  ```{bullet_summary}'''."
            summary = get_completion(prompt, MODEL_NAME)

            comment_summary.append({"content": summary, "topicKey": topic})
            json_object["summary"]["topics"].append({"key": topic, "name": topic})
            new_percentage = float(1/(num_cells))
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


def generate_user_analysis(topics, lens_id, whiteboard_id):
    try:
        start_time = time.time()
        print("topics", topics)
        topics = [clean_insight_area(topic) for topic in topics]
        topics = [topic for topic in topics if topic != ""]
        if topics:
            json_object = generate_from_existing_topics(topics, lens_id, whiteboard_id)
        else:
            json_object = generate_from_scratch(lens_id, whiteboard_id)

        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        return json_object
    except Exception as e:
        print(f"Error in task: {e}")
        update_whiteboard_status("error", whiteboard_id)


# topics = ["existing solutions and problems", "yodeai impressions and wants"]
# lens_id = 874
# print(generate_user_analysis(topics, lens_id))