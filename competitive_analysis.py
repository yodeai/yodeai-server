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
from processBlock import stuff_summary
import json

areas_of_analysis = [
    "integrations with third party apps/services",
    "core features",
    "use of artificial intelligence",
]
COMPANY_SUMMARY = "overall description of what the company does"
def embed_areas(areas_of_analysis):
    result = {}
    total_areas = areas_of_analysis + [COMPANY_SUMMARY]
    for area in total_areas:
        embedding=getEmbeddings(area)
        result[area] = embedding
    return result

areas_of_analysis_embedding = embed_areas(areas_of_analysis)
urls = {
    "https://craft.io/": "Craft.io",
    "https://airfocus.com/": "Airfocus",
    # "Miro":"https://miro.com/",
}
MAX_DEPTH = 1
MAX_HEAP_SIZE = 5


def get_page_content(url):
    loader = WebBaseLoader(url)
    doc= loader.load()
    document = doc[0] if doc else None
    return document.page_content if document else ""

def build_heap(parent_url, area):
    heap = []
    visited_urls = set()
    extended_urls = [(parent_url, 0)] # add depth information
    heapq.heapify(heap)
    visited_urls.add(parent_url)
    while extended_urls:
        current_url, current_depth = extended_urls.pop()
        print(f"url of {len(extended_urls)}", current_url)
        print("\n")
        document_content = get_page_content(current_url)
        # Check relevance score
        relevance_score = give_relevance_score(document_content, area)
        print("added the doc to heap", current_url)
        heapq.heappush(heap, (-relevance_score, current_url))
        
        # Ensure the heap size does not exceed MAX_HEAP_SIZE
        while len(heap) > MAX_HEAP_SIZE:
            heap = heapq.nsmallest(MAX_HEAP_SIZE, heap)
            heapq.heapify(heap)
        if current_depth + 1 <= MAX_DEPTH:
            new_links = extract_links(current_url, parent_url, current_depth, visited_urls)
            extended_urls.extend(new_links)
            visited_urls.update(link[0] for link in new_links)
    return list(heap)

def add_to_block_and_chunk(url, parent_url):
    data, count = supabaseClient.table('web_content_block').select('content').eq('url', url).execute()
    document_content = data[1]
    if not document_content:
        document_content = get_page_content(url)
        # if (len(document_content) > 8000):
        #     document_content = stuff_summary(document_content)
        #     print("Summarized", document_content)
        document_content = re.sub(r'[^\x00-\x7F]+', '', document_content)
        document_content = document_content.replace('\u0000', '')

        block_data = {
            "url": url,
            "parent_url": parent_url,
            "content": document_content,
        }
        data, count = supabaseClient.table("web_content_block").insert(block_data).execute()
        block_id = data[1][0]["block_id"]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )
        chunks = text_splitter.split_text(document_content)
        cleaned_chunks = [remove_invalid_surrogates(text) for text in chunks]
        embeddings = getEmbeddings(cleaned_chunks)
        if embeddings is None:
            raise Exception("Error in getting embeddings.")
        
        for idx, (chunk, embedding) in enumerate(zip(cleaned_chunks, embeddings)):
            if idx > 100:
                break
            print (f"Processing chunk {idx} of block {block_id}")
            
            # Creating a new row in chunks table for each split
            supabaseClient.table('web_content_chunk').insert({
                'block_id': block_id,
                'content': chunk,
                'embedding': embedding,  
                'chunk_type': 'split',  
                'url': url,
                "parent_url": parent_url
            }).execute()
                
def web_crawl(parent_url, area):
    heap = build_heap(parent_url, area)
    print("DANA HEAP", heap)
    for (_, url) in heap:
        # Add to block and chunk table if not already there
        add_to_block_and_chunk(url, parent_url)


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
        final_links = set()
        # Extract and print the href attribute from each anchor tag
        for link in links:
            # print("checking link: ", link)
            if isinstance(link, Tag) and "href" in link.attrs:
                href = link["href"]

                # Use urljoin to handle missing scheme
                full_url = urljoin(url, href)

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
    content = content[:4000]
    prompt = f"Please give an overall score from 0 to 100 on how relevant this content: ```{content}''' is to {area}. Remember, return only an integer from 0 to 100."    
    response = get_completion(prompt, "gpt-3.5-turbo")

    # Extract the last sequence of digits from the response
    try:
        score = re.findall(r'\b\d+\b', response)[-1]
        print("relevance score is: ", score)
        print("\n")
        return int(score)
    except (IndexError, ValueError):
        print("Unable to extract a numerical score from the response.")
        return 0
    
def get_context(area, parent_url):
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

def generate_cell(area, parent_url): 
    context, relevant_urls = get_context(area, parent_url)
    prompt = f"Can you summarize information about {area} for this company {parent_url} using this context: {context}"    
    
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

def generate_company_overview(parent_url):
    add_to_block_and_chunk(parent_url, parent_url)
    return generate_cell(COMPANY_SUMMARY, parent_url)


def generate_json_file(skip_web_crawl=False):
    output_data = []

    for parent_url in urls.keys():
        company_data = {
            "company": urls[parent_url],
            "data": []
        }
        company_summary, relevant_urls = generate_company_overview(parent_url)
        # Add summary data
        company_data["data"].append({
            "summary": company_summary,
            "sources": relevant_urls
        })

        for area in areas_of_analysis:
            # web crawl
            if not skip_web_crawl:
                web_crawl(parent_url, area)
            # top chunks
            cell_text, area_sources = generate_cell(area, parent_url)
            area_data = {
                area: cell_text,
                "sources": area_sources
            }
            company_data["data"].append(area_data)

        output_data.append(company_data)

    # Write the JSON to a file
    with open("/Users/dfeng21/Desktop/research/competitive_analysis.json", "w") as json_file:
        json.dump(output_data, json_file, indent=2)
generate_json_file()