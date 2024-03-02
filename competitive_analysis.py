import requests
from bs4 import BeautifulSoup, Tag
from utils import get_completion, getEmbeddings
from urllib.parse import urlparse
import re
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import urljoin
from DB import supabaseClient
import heapq
from utils import remove_invalid_surrogates
import json
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio
import concurrent.futures
from sklearn.cluster import KMeans
from painpoint_analysis import find_closest_5_chunks
######################## CONSTANTS + UTILS ############################
KMEANS = 'KMEANS'
COMPANY_SUMMARY = "summary"
MODEL_NAME = "gpt-4" # sometimes won't work, use 3.5
def embed_areas(areas_of_analysis):
    result = {}
    total_areas = [COMPANY_SUMMARY] + areas_of_analysis
    for area in total_areas:
        embedding=getEmbeddings(area)
        result[area] = embedding
    return result

MAX_DEPTH = 1
MAX_HEAP_SIZE = 5

def get_page_content(url):
    loader = WebBaseLoader(url)
    doc= loader.load()
    document = doc[0] if doc else None
    return document.page_content if document else ""
def extract_links_from_html(html):
    # Define a regex pattern to match <a> tags
    pattern = r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"'

    # Find all matches of the pattern in the HTML
    matches = re.findall(pattern, html)

    # Return the list of links extracted from <a> tags
    return matches


def extract_links(url, parent_url, current_depth, visited_urls):
    links = set()
    try:
        # Fetch the webpage content
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception if there's an error

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all anchor tags (links) in the page
        links = soup.find_all('a', href=True)
        soup_gathered = True
        if not links:
            print("could not find links in soup")
            links = extract_links_from_html(response.text)
            soup_gathered = False

        final_links = set()
        # Extract and print the href attribute from each anchor tag
        for link in links:
            # print("checking link: ", link)
            if soup_gathered:
                if isinstance(link, Tag) and "href" in link.attrs:
                    href = link["href"]
                    # Use urljoin to handle missing scheme
                    full_url = urljoin(url, href)
                else:
                    continue
            else:
                full_url = urljoin(url, link)

            if '#' in full_url:
                continue
            if not full_url.isspace() and len(full_url) > 0 and full_url not in visited_urls and full_url.startswith(parent_url):
                # Skip if the link is from a different subdomain
                if urlparse(full_url).netloc != urlparse(parent_url).netloc:
                    continue
                final_links.add((full_url, current_depth + 1))

        return list(final_links)

    except requests.exceptions.RequestException as e:
        print("Error fetching the webpage:", e)
        return []

def give_relevance_score(content, area): 
    summarized_content = content[:4000]
    prompt = f"Please give an overall score from 0 to 100 if the ```{summarized_content}''' describes the company's {area}. If the content directly mentions or paraphrases {area} in its text, then output 100. Remember, return only an integer from 0 to 100."    
    response = get_completion(prompt, "gpt-3.5-turbo")
    
    # Extract the last sequence of digits from the response
    try:
        score = re.findall(r'\b\d+\b', response)[-1]
        # print("relevance score is: ", score)
        print("\n")
        return int(score)
    except (IndexError, ValueError):
        # print("Unable to extract a numerical score from the response.")
        return 0
    
def get_context(area, parent_url, areas_of_analysis_embedding):
    rpc_params = {
        "matchcount": 5, 
        "queryembedding": areas_of_analysis_embedding[area],
        "parenturl": parent_url
    }
    data, error = supabaseClient.rpc("get_top_chunks_for_competitive_analysis", rpc_params).execute() 
    relevant_chunks = data[1]
    relevant_urls = set(d['url'] for d in relevant_chunks)
    text = ""    
    for d in relevant_chunks:        
        text += d['content'] + "\n\n"
    return text, list(relevant_urls)

def generate_cell(area, parent_url, areas_of_analysis_embedding): 
    context, relevant_urls = get_context(area, parent_url, areas_of_analysis_embedding)
    prompt = f"Please summarize information about {area} for this company {parent_url} using this context: {context}, and limit the summary to only 1 paragraph. IF THE CONTEXT IS NOT RELEVANT to {area}, THEN OUTPUT 'NO INFORMATION FOUND'"    
    
    response = get_completion(prompt, "gpt-3.5-turbo")

    return response, relevant_urls

def extract_website_address(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

def replace_two_whitespace(input_string):
    result_string = re.sub(r'(\s)\1+', r'\1', input_string)
    return result_string

def save_output_to_file(output, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(output)
###################### HEAP APPROACH #########################
# def generate_company_overview(parent_url):
#     add_to_block_and_chunk(parent_url, parent_url)
#     return generate_cell(COMPANY_SUMMARY, parent_url)

# def build_heap(parent_url, area):
#     heap = []
#     visited_urls = set()
#     extended_urls = [(parent_url, 0)] # add depth information
#     heapq.heapify(heap)
#     visited_urls.add(parent_url)
#     while extended_urls:
#         current_url, current_depth = extended_urls.pop()
#         print(f"url of {len(extended_urls)}", current_url)
#         print("\n")
#         document_content = get_page_content(current_url)
#         # Check relevance score
#         relevance_score = give_relevance_score(document_content, area)
#         print("added the doc to heap", current_url)
#         heapq.heappush(heap, (-relevance_score, current_url))
        
#         # Ensure the heap size does not exceed MAX_HEAP_SIZE
#         while len(heap) > MAX_HEAP_SIZE:
#             heap = heapq.nsmallest(MAX_HEAP_SIZE, heap)
#             heapq.heapify(heap)
#         if current_depth + 1 <= MAX_DEPTH:
#             new_links = extract_links(current_url, parent_url, current_depth, visited_urls)
#             extended_urls.extend(new_links)
#             visited_urls.update(link[0] for link in new_links)
#     return (heap)

# def add_to_block_and_chunk(url, parent_url):
#     data, count = supabaseClient.table('web_content_block').select('content').eq('url', url).execute()
#     document_content = data[1]
#     if not document_content:
#         document_content = get_page_content(url)
#         document_content = remove_invalid_surrogates(document_content)
#         # document_content = re.sub(r'[^\x00-\x7F]+', '', document_content)
#         # document_content = document_content.replace('\u0000', '')

#         block_data = {
#             "url": url,
#             "parent_url": parent_url,
#             "content": document_content,
#         }
#         data, count = supabaseClient.table("web_content_block").insert(block_data).execute()
#         block_id = data[1][0]["block_id"]
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=800,
#             chunk_overlap=200,
#             separators=["\n\n", "\n", "(?<=\. )", " ", ""]
#         )
#         chunks = text_splitter.split_text(document_content)
#         cleaned_chunks = [remove_invalid_surrogates(text) for text in chunks]
#         embeddings = getEmbeddings(cleaned_chunks)
#         if embeddings is None:
#             raise Exception("Error in getting embeddings.")
        
#         for idx, (chunk, embedding) in enumerate(zip(cleaned_chunks, embeddings)):
#             if idx > 100:
#                 break
#             print (f"Processing chunk {idx} of block {block_id}")
            
#             # Creating a new row in chunks table for each split
#             supabaseClient.table('web_content_chunk').insert({
#                 'block_id': block_id,
#                 'content': chunk,
#                 'embedding': embedding,  
#                 'chunk_type': 'split',  
#                 'url': url,
#                 "parent_url": parent_url
#             }).execute()

# def web_crawl(parent_url, area):
#     heap = build_heap(parent_url, area)
#     print("EXISTING HEAP", heap)
#     for (_, url) in heap:
#         # Add to block and chunk table if not already there
#         add_to_block_and_chunk(url, parent_url)


# ################################ RELEVANT CHUNKS METHOD ###########################
# def add_relevant_chunks(url, area, parent_url):
#     relevant = 0
#     # get page content
#     data, count = supabaseClient.table('web_content_block').select('content').eq('url', url).execute()
#     document_content = data[1]
#     if not document_content:
#         document_content = get_page_content(url)
#         document_content = re.sub(r'[^\x00-\x7F]+', '', document_content)
#         document_content = document_content.replace('\u0000', '')
#     else: 
#         document_content = document_content[0]["content"]

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=4000,
#         chunk_overlap=0,
#         separators=["\n\n", "\n", "(?<=\. )", " ", ""]
#     )
#     chunks = text_splitter.split_text(document_content)
#     cleaned_chunks = [remove_invalid_surrogates(text) for text in chunks]
#     idx = 0
#     for chunk_content in cleaned_chunks:
#         if idx >= 100:
#             break

#         if give_relevance_score(chunk_content, area) >= 100:
#             relevant += 1
#             if relevant == 1:
#                 # add to block if one relevant chunk
#                 block_data = {
#                     "url": url,
#                     "parent_url": parent_url,
#                     "content": document_content,
#                 }
#                 data, count = supabaseClient.table('web_content_block').select('block_id').eq('url', url).execute()
#                 block_id = data[1]
#                 if not block_id:
#                     data, count = supabaseClient.table("web_content_block").insert(block_data).execute()
#                     block_id = data[1][0]["block_id"]
#                 else:
#                     block_id = block_id[0]["block_id"]

#             # otherwise we already have block id
#             # store the chunks
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=800,
#                 chunk_overlap=200,
#                 separators=["\n\n", "\n", "(?<=\. )", " ", ""]
#             )
#             mini_chunks = text_splitter.split_text(chunk_content)
#             mini_cleaned_chunks = [remove_invalid_surrogates(text) for text in mini_chunks]
#             embeddings = getEmbeddings(mini_cleaned_chunks)
#             if embeddings is None:
#                 raise Exception("Error in getting embeddings.")
            
#             for (chunk, embedding) in zip(mini_cleaned_chunks, embeddings):
#                 if idx >= 100:
#                     break
#                 print (f"Processing chunk {idx} of block {block_id}")
#                 # Check if existing chunk
#                 data, count = supabaseClient.table('web_content_chunk').select('url').eq('content', chunk).execute()
#                 existing_url = data[1]
#                 if not existing_url:
#                     # Creating a new row in chunks table for each split
#                     supabaseClient.table('web_content_chunk').insert({
#                         'block_id': block_id,
#                         'content': chunk,
#                         'embedding': embedding,  
#                         'chunk_type': 'split',  
#                         'url': url,
#                         "parent_url": parent_url
#                     }).execute()
#                 idx += 1
    
#     return relevant > 0
    
# def web_crawl2(parent_url, area):
#     visited_urls = set()
#     extended_urls = [(parent_url, 0)] # add depth information
#     visited_urls.add(parent_url)
#     while extended_urls:
#         current_url, current_depth = extended_urls.pop()
#         print(f"url of {len(extended_urls)}", current_url)
#         print("\n")
#         # Check relevance score
#         added_relevance_chunks=add_relevant_chunks(current_url, area, parent_url)
#         if added_relevance_chunks:
#             print("added the doc to heap", current_url)

#         # Can edit this so that it stops at if there is relevant chunks or not (so do not cap at MAX_DEPTH)
#         if current_depth + 1 <= MAX_DEPTH:
#             new_links = extract_links(current_url, parent_url, current_depth, visited_urls)
#             extended_urls.extend(new_links)
#             visited_urls.update(link[0] for link in new_links)

# def generate_json_file(skip_web_crawl=False):
#     output_data = []
#     start_time = time.time()

#     for parent_url in urls.keys():
#         print("parent_url", parent_url)
#         company_data = {
#             "company": urls[parent_url],
#             "data": []
#         }
#         company_summary, relevant_urls = generate_company_overview(parent_url)
#         # Add summary data
#         company_data["data"].append({
#             "title": "summary",
#             "content": company_summary,
#             "sources": relevant_urls
#         })

#         for area in areas_of_analysis:
#             print("area", area)
#             # web crawl
#             if not skip_web_crawl:
#                 # web_crawl2(parent_url, area)
#                 web_crawl(parent_url, area)
#             # top chunks
#             cell_text, area_sources = generate_cell(area, parent_url)
#             area_data = {
#                 "title": area,
#                 "content": cell_text,
#                 "sources": area_sources
#             }
#             company_data["data"].append(area_data)

#         output_data.append(company_data)
#     print(f"Time taken: {time.time() - start_time:.2f} seconds")

#     # Write the JSON to a file
#     with open("/Users/dfeng21/Desktop/research/competitive_analysis2.json", "w") as json_file:
#         json.dump(output_data, json_file, indent=2)




############################ OPTMIZED HEAP APPROACH #############################
# def add_to_block_and_chunk(url, parent_url):
#     data, count = supabaseClient.table('web_content_block').select('content').eq('url', url).execute()
#     document_content = data[1]

#     if not document_content:
#         document_content = get_page_content(url)
#         document_content = remove_invalid_surrogates(document_content)

#         block_data = {
#             "url": url,
#             "parent_url": parent_url,
#             "content": document_content,
#         }

#         # Insert block data individually
#         data, count = supabaseClient.table("web_content_block").insert(block_data).execute()
#         block_id = data[1][0]["block_id"]

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=800,
#             chunk_overlap=200,
#             separators=["\n\n", "\n", "(?<=\. )", " ", ""]
#         )
#         chunks = text_splitter.split_text(document_content)
#         chunks = chunks[:100]
#         cleaned_chunks = [remove_invalid_surrogates(text) for text in chunks]
#         embeddings = getEmbeddings(cleaned_chunks)

#         if embeddings is None:
#             raise Exception("Error in getting embeddings.")

#         # Batch insert for chunks
#         with ThreadPoolExecutor() as executor:
#             formatted_chunks = list(executor.map(lambda args: {
#                 'block_id': block_id,
#                 'content': args[0],
#                 'embedding': args[1],
#                 'chunk_type': 'split',
#                 'url': url,
#                 "parent_url": parent_url
#             }, zip(cleaned_chunks, embeddings)))
#         data, count = supabaseClient.table('web_content_chunk').insert(formatted_chunks).execute()

# def generate_company_overview(parent_url, areas_of_analysis_embedding):
#     add_to_block_and_chunk(parent_url, parent_url)
#     return generate_cell(COMPANY_SUMMARY, parent_url, areas_of_analysis_embedding)

# def gather_depth_2_links(parent_url):    
#     visited_urls = set()
#     extended_urls = []
#     visited_urls.add(parent_url)
    
#     new_links = extract_links(parent_url, parent_url, 0, visited_urls)
#     extended_urls.extend(new_links)
#     visited_urls.update(link[0] for link in new_links)
    
#     # # Fetch and process links at depth 2 concurrently
#     with ThreadPoolExecutor() as executor:
#         new_links_depth_2 = list(executor.map(lambda url: extract_links(url, parent_url, 1, visited_urls), extended_urls))
    
#     visited_urls.update(link[0] for links in new_links_depth_2 for link in links)
    
#     return list(visited_urls)

# def build_heap_with_predefined_links(area, predefined_links):
#     heap = []
#     heapq.heapify(heap)
#     for i, current_url in enumerate(predefined_links):
#         print(f"url of {i}", current_url)
#         print("\n")
#         document_content = get_page_content(current_url)
#         # Check relevance score
#         relevance_score = give_relevance_score(document_content, area)
#         print("added the doc to heap", current_url)
#         heapq.heappush(heap, (-relevance_score, current_url))
#     return heap

# def web_crawl_optimized(parent_url, area, predefined_links):
#     heap = build_heap_with_predefined_links(area, predefined_links)
#     print("HEAP", heap)
#     for _ in range(MAX_HEAP_SIZE):
#         score, url = heapq.heappop(heap)
#         add_to_block_and_chunk(url, parent_url)


# def crawl_and_generate_data_wrapper(parent_url, urls, areas_of_analysis_embedding, skip_web_crawl=False):
#     result = crawl_and_generate_data(parent_url, urls, areas_of_analysis_embedding, skip_web_crawl)
#     print(f"Process for {parent_url} has finished.")
#     return result

# def crawl_and_generate_data(parent_url, urls, areas_of_analysis_embedding, skip_web_crawl=False):
#     if not skip_web_crawl:
#         predefined_links = gather_depth_2_links(parent_url)
#         print("LINKS", predefined_links)

#         with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#             # Submit tasks for each area concurrently
#             futures = [executor.submit(web_crawl_optimized, parent_url, area, predefined_links) for area in areas_of_analysis_embedding.keys()]

#             # Wait for all tasks to complete
#             concurrent.futures.wait(futures)
#             # for link in predefined_links:
#             #     add_to_block_and_chunk(link, parent_url)

#     print("Starting retrieval")
#     company_data = {
#         "company": urls[parent_url],
#         "data": []
#     }
#     company_summary, relevant_urls = generate_company_overview(parent_url, areas_of_analysis_embedding)
#     # Add summary data
#     company_data["data"].append({
#         "title": "summary",
#         "content": company_summary,
#         "sources": relevant_urls
#     })
#     for area in areas_of_analysis_embedding.keys():
#         area_data_list = generate_area_data(area, parent_url, areas_of_analysis_embedding)
#         company_data["data"].append(area_data_list)

#     return company_data

# def generate_area_data(area, parent_url, areas_of_analysis_embedding):
#     # Generate data for a specific area
#     cell_text, area_sources = generate_cell(area, parent_url, areas_of_analysis_embedding)
#     return {
#         "title": area,
#         "content": cell_text,
#         "sources": area_sources
#     }


# def generate_json_file_optimized(urls, areas_of_analysis, skip_web_crawl=False):
#     output_data = []
#     start_time = time.time()
#     areas_of_analysis_embedding = embed_areas(areas_of_analysis)

#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         # Submit tasks and store the Future objects
#         futures = [executor.submit(crawl_and_generate_data_wrapper, parent_url, urls.keys(), areas_of_analysis_embedding, skip_web_crawl) for parent_url in urls.keys()]

#         # Collect results from completed tasks
#         for future in concurrent.futures.as_completed(futures):
#             try:
#                 result = future.result()
#                 output_data.append(result)
#             except Exception as e:
#                 print(f"Error in task: {e}")

#     print(f"Time taken: {time.time() - start_time:.2f} seconds")

#     # Write the JSON to a file
#     with open("/Users/dfeng21/Desktop/research/competitive_analysis_optimized2.json", "w") as json_file:
#         json.dump(output_data, json_file, indent=2)
#     return output_data

############################## PROCESS EVERYTHING APPROACH ######################
def gather_depth_1_links(parent_url):    
    visited_urls = set()
    extended_urls = []
    visited_urls.add(parent_url)
    
    new_links = extract_links(parent_url, parent_url, 0, visited_urls)
    extended_urls.extend(new_links)
    visited_urls.update(link[0] for link in new_links)
    
    urls = list(visited_urls)[:50]
    return urls

def gather_depth_2_links(depth_1_links):    
    visited_urls = set()
    extended_urls = []
    for parent_url in depth_1_links:
        visited_urls.add(parent_url)
        
        new_links = extract_links(parent_url, parent_url, 0, visited_urls)
        extended_urls.extend(new_links)
        visited_urls.update(link[0] for link in new_links)
    
    urls = list(visited_urls)[:100]
    return urls

def add_to_block_and_chunk(url, parent_url):
    data, count = supabaseClient.table('web_content_block').select('content').eq('url', url).execute()
    document_content = data[1]

    if not document_content:
        document_content = get_page_content(url)
        document_content = remove_invalid_surrogates(document_content)

        block_data = {
            "url": url,
            "parent_url": parent_url,
            "content": document_content,
        }

        # Insert block data individually
        data, count = supabaseClient.table("web_content_block").insert(block_data).execute()
        block_id = data[1][0]["block_id"]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )
        chunks = text_splitter.split_text(document_content)
        chunks = chunks[:100]
        cleaned_chunks = [remove_invalid_surrogates(text) for text in chunks]
        embeddings = getEmbeddings(cleaned_chunks)

        if embeddings is None:
            raise Exception("Error in getting embeddings.")

        # Batch insert for chunks
        with ThreadPoolExecutor() as executor:
            formatted_chunks = list(executor.map(lambda args: {
                'block_id': block_id,
                'content': args[0],
                'embedding': args[1],
                'chunk_type': 'split',
                'url': url,
                "parent_url": parent_url
            }, zip(cleaned_chunks, embeddings)))
        data, count = supabaseClient.table('web_content_chunk').insert(formatted_chunks).execute()

def generate_company_overview(parent_url, areas_of_analysis_embedding):
    add_to_block_and_chunk(parent_url, parent_url)
    return generate_cell(COMPANY_SUMMARY, parent_url, areas_of_analysis_embedding)

def crawl_for_data(parent_url, urls, num_urls, whiteboard_id, new_percentage, skip_web_crawl=False):
    if not skip_web_crawl:
        predefined_links = gather_depth_1_links(parent_url)
        data, error = supabaseClient.rpc("update_plugin_progress", {"id": whiteboard_id, "new_progress": new_percentage}).execute() 
        predefined_links += gather_depth_2_links(predefined_links)
        data, error = supabaseClient.rpc("update_plugin_progress", {"id": whiteboard_id, "new_progress": new_percentage}).execute() 
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit tasks for each area concurrently
            futures = [executor.submit(add_to_block_and_chunk, link, parent_url) for link in predefined_links]

            # Wait for all tasks to complete
            concurrent.futures.wait(futures)
        data, error = supabaseClient.rpc("update_plugin_progress", {"id": whiteboard_id, "new_progress": new_percentage}).execute() 



def generate_data(parent_url, urls, areas_of_analysis_embedding, num_urls, whiteboard_id, new_percentage, skip_web_crawl=False):
    company_data = {
        "company": urls[parent_url],
        "company_url": parent_url,
        "data": []
    }
    for area in areas_of_analysis_embedding.keys():
        area_data_list = generate_area_data(area, parent_url, areas_of_analysis_embedding)
        company_data["data"].append(area_data_list)
        # TODO: update progress since cell done
        data, error = supabaseClient.rpc("update_plugin_progress", {"id": whiteboard_id, "new_progress": new_percentage}).execute() 


    
    # Delete all rows from web_content_block with the specified parent_url
    supabaseClient.table('web_content_block').delete().eq('parent_url', parent_url).execute()
    print(f"Process for {parent_url} has finished.")

    return company_data

def generate_area_data(area, parent_url, areas_of_analysis_embedding):
    # Generate data for a specific area
    cell_text, area_sources = generate_cell(area, parent_url, areas_of_analysis_embedding)
    return {
        "title": area,
        "content": cell_text,
        "sources": area_sources
    }

def update_whiteboard_status(status, whiteboard_id):
    # Get the plugin
    data, _ = supabaseClient.table('whiteboard')\
        .select('plugin')\
        .eq('whiteboard_id', whiteboard_id)\
        .execute()
    json_object = data[1][0]["plugin"]
    json_object["state"]["status"] = status
    if status == "processing":
        json_object["state"]["progress"] = 0

    # Update the status of the block
    update_response, update_error = supabaseClient.table('whiteboard')\
        .update({'plugin': json_object})\
        .eq('whiteboard_id', whiteboard_id)\
        .execute()

def update_whiteboard_nodes(data, whiteboard_id):
    update_response, update_error = supabaseClient.table('whiteboard')\
    .update({'nodes': data})\
    .eq('whiteboard_id', whiteboard_id)\
    .execute()

def update_competitive_analysis(urls, areas_of_analysis, whiteboard_id, skip_web_crawl=False):
    additional_output_data = create_competitive_analysis(urls, areas_of_analysis, whiteboard_id, skip_web_crawl)
    # get existing data
    output_data, update_error = supabaseClient.table('whiteboard')\
    .select('nodes')\
    .eq('whiteboard_id', whiteboard_id)\
    .execute()
    output_data.extend(additional_output_data)
    return output_data
def get_chunks_from_urls(urls):
    chunks, count = supabaseClient.table('web_content_chunk').select("*").in_('parent_url', urls).execute()
    mapping = {}
    for chunk in chunks[1]:
        mapping[chunk['embedding']] = chunk
    return mapping
def clean_text(text):
    # Remove non-ASCII characters
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text)
    return cleaned_text
def generate_areas_of_analysis(urls, whiteboard_id, new_percentage, method=KMEANS):
    start_time = time.time()
    if method == KMEANS:
        chunks_mapping = get_chunks_from_urls(urls)
        chunks_list_str = list(chunks_mapping.keys())
        chunks_list = [json.loads(chunk_emb) for chunk_emb in chunks_list_str]
        kmeans = KMeans(n_clusters=min(3, len(chunks_list)), random_state=42, n_init=10)
        kmeans.fit(chunks_list)
        # Get cluster assignments for each chunk
        cluster_assignments = kmeans.labels_
        # Get centroid:
        cluster_centroids = kmeans.cluster_centers_
        # Collect chunks belonging to each cluster
        cluster_chunks = {}
        for i, assignment in enumerate(cluster_assignments):
            if assignment not in cluster_chunks:
                cluster_chunks[assignment] = []
            emb = chunks_list_str[i]
            cluster_chunks[assignment].append(chunks_mapping[emb])

        # Get closest 5 chunks to the cluster centroid and then extract topics from that
        topics = []
        for i, (cluster, chunks) in enumerate(cluster_chunks.items()):
            closest_chunks = find_closest_5_chunks(cluster_centroids[i], chunks)
            text = ""
            for d in closest_chunks:        
                text += d[0]['content'] + "\n\n"
                # give it a role: you are a business analyst pick areas of analysis that will allow you to compare 
            text = clean_text(text)
            prompt = f"You are a product manager that is trying to create a new product. Please output one main feature from this content: {str(text)} that is useful for competitive analysis between companies. OUTPUT THE TOPIC IN 4-5 WORDS ONLY. OUTPUT 'ERROR WITH GENERATING' IF THE TOPIC HAS ANYTHING TO DO WITH HTTP RESPONSE STATUS CODES AND NOT BEING ABLE TO CONNECT TO THE SITE."
            topic = get_completion(prompt, MODEL_NAME)
            cleaned_topic = topic.lower().replace('*', '')
            print("CLEAN", cleaned_topic)
            topics.append(cleaned_topic)
        data, error = supabaseClient.rpc("update_plugin_progress_spreadsheet", {"id": whiteboard_id, "new_progress": new_percentage}).execute() 
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        return topics

def create_competitive_analysis(urls, areas_of_analysis, whiteboard_id, skip_web_crawl=False):
    try:
        output_data = []
        start_time = time.time()
        update_whiteboard_status("processing", whiteboard_id)
        # competitive analysis progress calculation
        num_urls = len(urls)
        areas_of_analysis = [area for area in areas_of_analysis if area != ""]
        if not areas_of_analysis:
            new_percentage = 1/(num_urls*3 + num_urls*4 + 1) # web crawl, add to block table, cell generation (including summary), generate areas
        else:
            new_percentage = 1/(num_urls*3 + num_urls*(len(areas_of_analysis) + 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit tasks and store the Future objects
            futures = [executor.submit(crawl_for_data, parent_url, urls, num_urls, whiteboard_id, new_percentage, skip_web_crawl) for parent_url in urls.keys()]

            # Collect results from completed tasks
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Error in task for web crawling: {e}")
                    update_whiteboard_status("error", whiteboard_id)
                    raise
        
        if not areas_of_analysis:
            areas_of_analysis = generate_areas_of_analysis(urls, whiteboard_id, new_percentage)

        areas_of_analysis_embedding = embed_areas(areas_of_analysis)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit tasks and store the Future objects
            futures = [executor.submit(generate_data, parent_url, urls, areas_of_analysis_embedding, num_urls, whiteboard_id, new_percentage, skip_web_crawl) for parent_url in urls.keys()]

            # Collect results from completed tasks
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    output_data.append(result)
                except Exception as e:
                    print(f"Error in task for analysis generation: {e}")
                    update_whiteboard_status("error", whiteboard_id)
                    raise

        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        return output_data
    except Exception as e:
        print(f"Exception occurred while generating competitive analysis: {e}")
        update_whiteboard_status("error", whiteboard_id)
        raise


############################## METHOD TO RUN #########################
if __name__ == '__main__':
    # areas_of_analysis = [
    #     "integrations with third party apps/services",
    #     "core features",
    #     "use of artificial intelligence",
    # ]

    # urls = {
    #     # "https://www.salesforce.com/ap/products/einstein-ai-solutions/" : "Salesforce",
    #     # "https://www.kraftful.com/" : "Kraftful",
    #     "https://craft.io/": "Craft.io",
    #     "https://airfocus.com/": "Airfocus",
        
    # }
    # create_competitive_analysis(urls, areas_of_analysis, 160)
    pass