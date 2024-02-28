from user_analysis import get_block_content
from utils import get_completion
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

    description = ""
    for field in chosen_fields:
        if field == "Acceptance criteria/scope":
            description += "- **Acceptance criteria/scope:** \n\n"
            description += generate_acceptance_criteria(prd, title)
        elif field == "Background":
            description += "- **Background:** \n\n"
            description += generate_background(prd, title)
        elif field == "Priority level":
            description += "- **Priority level:** \n\n"
            description += generate_priority_level(prd, title)
        elif field == "In and out of scope":
            description += "- **In and out of scope:** \n\n"
            description += generate_in_out_scope(prd, title)
        elif field == "User story":
            description += "- **User story:** \n\n"
            description += generate_user_story(prd, title)
        elif field == "Technical considerations":
            description += "- **Technical considerations:** \n\n"
            description += generate_technical_considerations(prd, title)
        elif field == "Design field":
            description += "- **Design field:** \n\n"
            description += generate_design_field(prd, title)
        elif field == "Questions":
            description += "- **Questions:** \n\n"
            description += generate_questions(prd, title)
    print(description)

    return {"title": title, "project_id": project_id, "summary": summary, "description": description}

def generate_acceptance_criteria(prd, title):
    prompt = f"Generate acceptance criteria/scope that is based on this topic: {title} in this product requirement document: {prd}."
    response = get_completion(prompt, MODEL_NAME)
    return response

def generate_background(prd, title):
    prompt = f"Generate background information is relevant for this ticket based on this topic: {title} in this product requirement document: {prd}."
    response = get_completion(prompt, MODEL_NAME)
    return response

def generate_priority_level(prd, title):
    prompt = f"Define the priority level is relevant for this ticket based on this topic: {title} in this product requirement document: {prd}."
    response = get_completion(prompt, MODEL_NAME)
    return response

def generate_in_out_scope(prd, title):
    prompt = f"Describe what's in scope and what's out of scope that is relevant for this ticket based on this topic: {title} in this product requirement document: {prd}."
    response = get_completion(prompt, MODEL_NAME)
    return response

def generate_user_story(prd, title):
    prompt = f"Write a user story is relevant for this ticket based on this topic: {title} in this product requirement document: {prd}."
    response = get_completion(prompt, MODEL_NAME)
    return response

def generate_technical_considerations(prd, title):
    prompt = f"Discuss technical considerations is relevant for this ticket based on this topic: {title} in this product requirement document: {prd}."
    response = get_completion(prompt, MODEL_NAME)
    return response

def generate_design_field(prd, title):
    prompt = f"Provide UX details and reference mockups is relevant for this ticket based on this topic: {title} in this product requirement document: {prd}."
    response = get_completion(prompt, MODEL_NAME)
    return response

def generate_questions(prd, title):
    prompt = f"List any questions or concerns is relevant for this ticket based on this topic: {title} in this product requirement document: {prd}."
    response = get_completion(prompt, MODEL_NAME)
    return response

def generate_summary(prd, title):
    prompt = f"I want to generate the summary of this ticket based on this topic: {title} in this product requirement document: {prd}. Remember, the summary should describe value for business and customers, not the features of the ticket, and should be less than 100 words."   
    response = get_completion(prompt, MODEL_NAME)
    # Process the response as needed
    return response

def create_jira_tickets(prd_block_id, project_id, chosen_fields):
    start_time = time.time()
    content = get_block_content(prd_block_id)
    topics = generatePossibleTicketTopics(content)
    topics = topics[:4]
    result = []
    for topic in topics:
        ticket = generateATicket(topic, chosen_fields, project_id, content)
        result.append(ticket)
    file_path = 'jira_tickets.json'

    # Serialize the list of dictionaries to JSON and write it to the file
    with open(file_path, 'w') as json_file:
        json.dump(result, json_file, indent=4)
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

    return result




livechat = 2978
yodeai = 2981
airpods = 2979
product_hunt = 2980


# project_id = -1
# chosen_fields = ["Acceptance criteria/scope", "Background", "Priority level", "In and out of scope","User story", "Technical considerations", "Design field", "Questions"]
# create_jira_tickets(product_hunt, project_id, chosen_fields)