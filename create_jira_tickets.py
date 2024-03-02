from user_analysis import get_block_content
from utils import get_completion, supabaseClient
import re
import json
import time

MODEL_NAME = "gpt-3.5-turbo"

def splitTicketTopics(input_string):
    # Splitting the string using "\n" and "."
    result = re.split(r'[\n.]', input_string)

    # Removing empty strings from the result
    result = [part.strip() for part in result if (len(part.strip())>3)]

    splits=[]
    for i, part in enumerate(result, 1):
        splits.append(part)
    return splits

def generatePossibleTicketTopics(prd):
    prompt = f"I want break up this product requirement document into Jira tasks:``{prd}'', please output a list of no more than 8 task titles, that are no longer than 4-5 phrases, and are specific actionable jira issues."   
    response = get_completion(prompt, MODEL_NAME)
    return splitTicketTopics(response)

def generateATicket(title, chosen_fields, project_id, prd):
    print("Title")
    print(title)
    print("\n\n")

    summary = generate_summary(prd, title)

    print("Summary")
    print(summary)
    print("\n\n")
    
    print("Description")

    # description = ""
    # for field in chosen_fields:
    #     if field == "Acceptance criteria/scope":
    #         description += "**Acceptance criteria/scope:** \n\n"
    #         description += generate_acceptance_criteria(prd, title)
    #     elif field == "Background":
    #         description += "**Background:** \n\n"
    #         description += generate_background(prd, title)
    #     elif field == "Priority level":
    #         description += "**Priority level:** \n\n"
    #         description += generate_priority_level(prd, title)
    #     elif field == "In and out of scope":
    #         description += "**In and out of scope:** \n\n"
    #         description += generate_in_out_scope(prd, title)
    #     elif field == "User story":
    #         description += "**User story:** \n\n"
    #         description += generate_user_story(prd, title)
    #     elif field == "Technical considerations":
    #         description += "**Technical considerations:** \n\n"
    #         description += generate_technical_considerations(prd, title)
    #     elif field == "Design field":
    #         description += "**Design field:** \n\n"
    #         description += generate_design_field(prd, title)
    #     elif field == "Questions":
    #         description += "**Questions:** \n\n"
    #         description += generate_questions(prd, title)
    description = generate_whole_jira_description(prd, chosen_fields, title)
    print(description)

    return {"title": title, "project_id": project_id, "summary": summary, "description": description}

def generate_acceptance_criteria(prd, title):
    prompt = f"Generate acceptance criteria/scope that is based on this topic: {title} in this product requirement document: {prd}."
    response = get_completion(prompt, MODEL_NAME)
    return response + "\n\n"

def generate_background(prd, title):
    prompt = f"Generate background information is relevant for this ticket based on this topic: {title} in this product requirement document: {prd}."
    response = get_completion(prompt, MODEL_NAME)
    return response + "\n\n"

def generate_priority_level(prd, title):
    prompt = f"Define the priority level is relevant for this ticket based on this topic: {title} in this product requirement document: {prd}."
    response = get_completion(prompt, MODEL_NAME)
    return response + "\n\n"

def generate_in_out_scope(prd, title):
    prompt = f"Describe what's in scope and what's out of scope that is relevant for this ticket based on this topic: {title} in this product requirement document: {prd}."
    response = get_completion(prompt, MODEL_NAME)
    return response + "\n\n"

def generate_user_story(prd, title):
    prompt = f"Write a user story is relevant for this ticket based on this topic: {title} in this product requirement document: {prd}."
    response = get_completion(prompt, MODEL_NAME)
    return response + "\n\n"

def generate_technical_considerations(prd, title):
    prompt = f"Discuss technical considerations is relevant for this ticket based on this topic: {title} in this product requirement document: {prd}."
    response = get_completion(prompt, MODEL_NAME)
    return response + "\n\n"

def generate_design_field(prd, title):
    prompt = f"Provide UX details and reference mockups is relevant for this ticket based on this topic: {title} in this product requirement document: {prd}."
    response = get_completion(prompt, MODEL_NAME)
    return response + "\n\n"

def generate_questions(prd, title):
    prompt = f"List any questions or concerns is relevant for this ticket based on this topic: {title} in this product requirement document: {prd}."
    response = get_completion(prompt, MODEL_NAME)
    return response + "\n\n"

def generate_summary(prd, title):
    prompt = f"I want to generate the summary of this ticket based on this topic: {title} in this product requirement document: {prd}. Remember, the summary should describe value for business and customers, not the features of the ticket, and should be less than 100 words."   
    response = get_completion(prompt, MODEL_NAME)
    # Process the response as needed
    return response

def generate_whole_jira_description(prd, fields, title):
    prompt = f"You are a product manager generating a Jira ticket description, where the ticket is focused on: {title} in this product requirement document: {prd}. Please include the following fields when generating the description, and bold each field: {', '.join(fields)}. Do not output a title."   
    response = get_completion(prompt, MODEL_NAME)
    # Process the response as needed
    return response

def update_widget_status(status, widget_id):
    # Get the plugin
    data, _ = supabaseClient.table('widget')\
        .select('state')\
        .eq('widget_id', widget_id)\
        .execute()
    json_object = data[1][0]["state"]
    json_object["status"] = status
    if status == "processing":
        json_object["progress"] = 0

    # Update the status of the block
    update_response, update_error = supabaseClient.table('widget')\
        .update({'state': json_object})\
        .eq('widget_id', widget_id)\
        .execute()

def update_widget_progress(progress, widget_id):
    # Get the plugin
    data, _ = supabaseClient.table('widget')\
        .select('state')\
        .eq('widget_id', widget_id)\
        .execute()
    json_object = data[1][0]["state"]
    print("json_object", json_object)
    json_object["progress"] = json_object["progress"] + progress

    # Update the status of the block
    update_response, update_error = supabaseClient.table('widget')\
        .update({'state': json_object})\
        .eq('widget_id', widget_id)\
        .execute()
    
def update_widget_nodes(data, widget_id):
    data = {"tickets": data}
    update_response, update_error = supabaseClient.table('widget')\
    .update({'output': data})\
    .eq('widget_id', widget_id)\
    .execute()

def create_jira_tickets(widget_id, prd_block_id, project_id, chosen_fields):
    start_time = time.time()
    update_widget_status("processing", widget_id)
    content = get_block_content(prd_block_id)
    new_percentage = 1/5
    topics = generatePossibleTicketTopics(content)
    # update_widget_progress(new_percentage, widget_id)
    data, error = supabaseClient.rpc("update_plugin_progress_widget", {"id": widget_id, "new_progress": new_percentage}).execute() 

    topics = topics[:4]
    result = []
    for topic in topics:
        ticket = generateATicket(topic, chosen_fields, project_id, content)
        result.append(ticket)
        # update_widget_progress(new_percentage, widget_id)
        data, error = supabaseClient.rpc("update_plugin_progress_widget", {"id": widget_id, "new_progress": new_percentage}).execute() 

    # file_path = 'jira_tickets.json'

    # Serialize the list of dictionaries to JSON and write it to the file
    # with open(file_path, 'w') as json_file:
    #     json.dump(result, json_file, indent=4)
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

    return result




livechat = 2978
yodeai = 2981
airpods = 2979
product_hunt = 2980


# project_id = -1
# chosen_fields = ["Acceptance criteria/scope", "Background", "Priority level", "In and out of scope","User story", "Technical considerations", "Design field", "Questions"]
# create_jira_tickets(product_hunt, project_id, chosen_fields)