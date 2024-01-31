from DB import supabaseClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import remove_invalid_surrogates, get_completion
import time
from competitive_analysis import update_whiteboard_status

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
    prompt = f"This content comes from a user interview, please return background about this interviewee:```{content}'''"
    return get_completion(prompt, MODEL_NAME)

def generate_user_analysis(topics, lens_id, whiteboard_id):
    try:
        start_time = time.time()
        update_whiteboard_status("processing", whiteboard_id)
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
                print("topic", topic)
                bullet_summary = ""
                for chunk_id, chunk_content in enumerate(cleaned_chunks):
                    prompt = f"Please output one bullet point summary of:  ```{chunk_content}''' that relates to {topic}, where each bullet point starts with a '-', AND PLEASE LIMIT TO 1 BULLET POINT."
                    response = get_completion(prompt, MODEL_NAME)
                    bullet_summary += response

                comments = {"comments": [{"id": i, "comment": bullet} for i, bullet in enumerate(bullet_summary.split("- ")[:-1]) if bullet != ""],
                            "topicKey": topic, "topicName": topic}
                current_insights["data"].append(comments)

                prompt = f"Please output a maximum of a 100 word summary of these bulletted chunks:  ```{bullet_summary}'''."
                summary = get_completion(prompt, MODEL_NAME)

                comment_summary.append({"content": summary, "topicKey": topic})
                new_percentage = float(1/(num_cells))
                data, error = supabaseClient.rpc("update_plugin_progress", {"id": whiteboard_id, "new_progress": new_percentage}).execute() 

            json_object["insights"].append(current_insights)
            json_object["summary"]["users"].append({"id": user_id, "name": name, "commentSummary": comment_summary})
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        return json_object
    except Exception as e:
        print(f"Error in task: {e}")
        update_whiteboard_status("error", whiteboard_id)


# topics = ["existing solutions and problems", "yodeai impressions and wants"]
# lens_id = 874
# print(generate_user_analysis(topics, lens_id))