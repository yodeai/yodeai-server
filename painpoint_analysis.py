from DB import supabaseClient
from sklearn.cluster import KMeans, DBSCAN as db
import json
from utils import getEmbeddings, get_completion
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re 
from datetime import datetime
import csv
import time

MODEL_NAME = "gpt-4"
NUM_CLUSTERS = 5
# LENS_ID = 972
LENS_ID = 959
KMEANS = 'KMEANS'
DEFAULT="DEFAULT"
PAINPOINT="PAINPOINT"
eps = 0.15
# DBSCAN = 'DBSCAN'
# MIXTURE_OF_GAUSSIANS = "MOG"
# AGGLOMERATIVE_CLUSTERING = "AG"
# default_painpoints = ['Inconsistent and limited AI responses.', 'System crashes and unavailability.', 'Login loop and server crashing.', 'Limited features in paid version.', 'Incomplete message generation.']
default_painpoints = ['Buggy, glitchy, and keeps crashing', 'Removal of community feature', 'Too many costs', 'Undo/crashing issues after update', 'Paywalls for basic features'] # regular topic clustering
# painpoint clustering: ['**Paid features**', 'Community and content lost', '**Art lost due to app glitch**', 'Unwanted app changes', '**Ineffective app**']

def update_spreadsheet_status(status, spreadsheet_id):
    # Get the plugin
    data, _ = supabaseClient.table('spreadsheet')\
        .select('plugin')\
        .eq('spreadsheet_id', spreadsheet_id)\
        .execute()
    json_object = data[1][0]["plugin"]
    json_object["state"]["status"] = status
    if status == "processing":
        json_object["state"]["progress"] = 0.0

    # Update the status of the block
    update_response, update_error = supabaseClient.table('spreadsheet')\
        .update({'plugin': json_object})\
        .eq('spreadsheet_id', spreadsheet_id)\
        .execute()
    
def get_block_ids(lens_id):
    block_ids, count = supabaseClient.table('lens_blocks').select("block_id").eq('lens_id', lens_id).execute()
    return block_ids[1]

def get_block_content(block_id):
    block_content, count = supabaseClient.table('block').select('title', 'content').eq('block_id', block_id).execute()
    return block_content[1]

def get_chunks_from_block_ids(block_ids):
    block_ids_list = [block_info['block_id'] for block_info in block_ids]
    chunks, count = supabaseClient.table('chunk').select('content', 'embedding', 'block_id').in_('block_id', block_ids_list).execute()
    # return a mapping of the embedding to the content
    mapping = {}
    for chunk in chunks[1]:
        mapping[chunk['embedding']] = chunk
    return mapping

def cosine_similarity_vectors(vec1, vec2):
    # Ensure the vectors are not empty
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    # Calculate cosine similarity
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    return similarity

def find_closest_5_chunks(cluster_centroid, chunk_embeddings, supabase=True):
    # Calculate cosine similarity between the cluster centroid and each chunk embedding
    if supabase:
        similarities = [cosine_similarity_vectors(cluster_centroid, json.loads(emb["embedding"])) for emb in chunk_embeddings]
    else:
        similarities = [cosine_similarity_vectors(cluster_centroid, emb) for emb in chunk_embeddings]
    
    # Sort chunks based on similarity scores (in descending order)
    sorted_chunks = sorted(zip(chunk_embeddings, similarities), key=lambda x: x[1], reverse=True)
    
    # Select top 5 chunks
    closest_chunks = sorted_chunks[:5]
    
    return closest_chunks

def getPainPoints(review):
    prompt = f"I want to extract the main pain points that the user is facing from the review I provide below. Create a list of these pain points. Each item on the list should focus on a specific pain point that the user mentioned. Start the description of each item on the list with a short expressive name that summarizes the theme of that pain point. Remember, only list pain points, not positive comments from the user.  User Review: ``{review}''"   
    response = get_completion(prompt, MODEL_NAME)
    return response

def splitReviewPainpoints(input_string):
    # Splitting the string using "\n" and "."
    result = re.split(r'[\n.]', input_string)

    # Removing empty strings from the result
    result = [part.strip() for part in result if (len(part.strip())>3)]

    splits=[]
    for i, part in enumerate(result, 1):
        splits.append(part)
    return splits

def cluster_painpoints_for_topics(lens_id, num_topics=NUM_CLUSTERS, method=KMEANS):
    start_time = time.time()
    block_ids = get_block_ids(lens_id)
    painpoints = []
    for block_id in block_ids:
        block_content = get_block_content(block_id['block_id'])
        painpoint_list = getPainPoints(block_content)
        painpoints.extend(splitReviewPainpoints(painpoint_list))
    painpoint_embeddings_list = [getEmbeddings(painpoint) for painpoint in painpoints]
    painpoint_embeddings = {str(painpoint_embeddings_list[i]): painpoint for i, painpoint in enumerate(painpoints)}
    if method == KMEANS:
        kmeans = KMeans(n_clusters=num_topics, random_state=42, n_init=10)
        kmeans.fit(painpoint_embeddings_list)
        # Get cluster assignments for each chunk
        cluster_assignments = kmeans.labels_
        # Get centroid:
        cluster_centroids = kmeans.cluster_centers_
        cluster_chunks = {}
        for i, assignment in enumerate(cluster_assignments):
            if assignment not in cluster_chunks:
                cluster_chunks[assignment] = []
            cluster_chunks[assignment].append(painpoint_embeddings_list[i])
        # Get closest 5 chunks to the cluster centroid and then extract topics from that
        topics = []
        for i, (cluster, chunks) in enumerate(cluster_chunks.items()):
            closest_chunks = find_closest_5_chunks(cluster_centroids[i], chunks, False)
            text = ""
            for emb in closest_chunks:  
                text += painpoint_embeddings[str(emb[0])] + "\n\n"  
            prompt = f"Please summarize these similar pain points: {text} into one main painpoint. OUTPUT THE PAIN POINT IN 4-5 WORDS ONLY, AND MAKE SURE TO RETURN A PAINPOINT AND NOT AN EMPTY STRING"
            topic = get_completion(prompt, MODEL_NAME)
            topics.append(topic)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        return topics

def cluster_for_topics(lens_id, new_percentage, spreadsheet_id, num_clusters=NUM_CLUSTERS, method=KMEANS):
    start_time = time.time()
    block_ids = get_block_ids(lens_id)
    chunks_mapping = get_chunks_from_block_ids(block_ids)
    chunks_list_str = list(chunks_mapping.keys())
    chunks_list = [json.loads(chunk_emb) for chunk_emb in chunks_list_str]
    if method == KMEANS:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
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
            prompt = f"Please output one main pain point summarized from this collection of user reviews on a product: {text}. OUTPUT THE PAIN POINT IN 4-5 WORDS ONLY."
            topic = get_completion(prompt, MODEL_NAME)
            topics.append(topic)
        data, error = supabaseClient.rpc("update_plugin_progress_spreadsheet", {"id": spreadsheet_id, "new_progress": new_percentage}).execute() 
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        return topics
def update_spreadsheet_nodes(output, spreadsheet_id):
    update_response, update_error = supabaseClient.table('spreadsheet')\
    .update({'dataSource': output})\
    .eq('spreadsheet_id', spreadsheet_id)\
    .execute()

def relation_score(review, painpoint):
    prompt = f"I need to evaluate whether the user complains about the paint point '{painpoint}' in the following review. To evaluate, please assign a relevance score between 1 to 10, with 10 indicating that the pain point is highly relevant to what the user complains about in the review, and 1 indicating that the pain point is not relevant. Review: '{review}' <<Remember: Your response should be a single number, from 1 to 10, where 10 indicates the highest relevance score. >>>"
    response = get_completion(prompt, MODEL_NAME)
    try:
        score = re.findall(r'\b\d+\b', response)[-1]
        # print("relevance score is: ", score)
        print("\n")
        return int(score)
    except (IndexError, ValueError):
        # print("Unable to extract a numerical score from the response.")
        return 0

def cluster_reviews(lens_id, painpoints, spreadsheet_id, num_clusters, method=KMEANS):
    update_spreadsheet_status("processing", spreadsheet_id)
    if painpoints:
        new_percentage = float(1/(len(painpoints)))
    elif not painpoints:
        new_percentage = float(1/(len(painpoints) + num_clusters))
        print("num clusters", num_clusters)
        painpoints = cluster_for_topics(lens_id, new_percentage, spreadsheet_id, num_clusters)
    start_time = time.time()
    painpoint_embeddings = [getEmbeddings(painpoint) for painpoint in painpoints]
    block_ids = get_block_ids(lens_id)
    chunks_mapping = get_chunks_from_block_ids(block_ids)
    chunks_list_str = list(chunks_mapping.keys())
    chunks_list = [json.loads(chunk_emb) for chunk_emb in chunks_list_str]
        
    if method == 'KMEANS':
        kmeans = KMeans(n_clusters=len(painpoints), init=painpoint_embeddings, n_init=len(painpoints))
        # Get cluster assignments for each chunk
        kmeans.fit_predict(chunks_list)
        cluster_assignments = kmeans.labels_
        cluster_centroids = kmeans.cluster_centers_
        # Collect chunks belonging to each cluster
        cluster_chunks = {}
        for i, assignment in enumerate(cluster_assignments):
            if assignment not in cluster_chunks:
                cluster_chunks[assignment] = []
            emb = chunks_list_str[i]
            cluster_chunks[assignment].append(chunks_mapping[emb])
        painpoint_to_block_id = {}
        dates = set()
        for cluster, chunks in cluster_chunks.items():
            # Choose the representative pain point for the cluster
            representative_painpoint = painpoints[cluster]
            print("Painpoint: ", representative_painpoint)
            block_ids = set()
            # Associate the cluster with the representative pain point
            painpoint_to_block_id[representative_painpoint] = {}
            for chunk in chunks:
                # Exclude nodes from clusters if their similarity to center is not large enough
                embedding = json.loads(chunk['embedding'])
                if cosine_similarity_vectors(cluster_centroids[cluster], embedding) >= 1-eps:
                    # Also exclude nodes who do not pass the relation threshold
                    if relation_score(chunk['content'], painpoints[cluster]) >= 8:
                        block_ids.add(chunk['block_id'])
                        # get block date
                        data, error = supabaseClient.table('block').select('original_date').eq('block_id', chunk['block_id']).execute()
                        original_date = data[1][0]['original_date']
                        date_obj = datetime.strptime(original_date, "%Y-%m-%d")
                        month_year = date_obj.strftime("%m/%Y")

                        if month_year not in painpoint_to_block_id[representative_painpoint]:
                            painpoint_to_block_id[representative_painpoint][month_year] = []
                            dates.add(month_year)
                        painpoint_to_block_id[representative_painpoint][month_year].append(chunk['block_id'])
                        print("Block: ", chunk['block_id'])
            print("\n")
        data, error = supabaseClient.rpc("update_plugin_progress_spreadsheet", {"id": spreadsheet_id, "new_progress": new_percentage}).execute() 
        
        result = convert_data(painpoint_to_block_id, dates)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        return result
def custom_sort(month):
    # Split month and year
    month_str, year_str = month.split("/")
    
    # Convert month and year to integers
    month_num = int(month_str)
    year_num = int(year_str)
    
    # Return a tuple of (year, month) for sorting
    return (year_num, month_num)
def convert_date(date_str):
    # Convert the date string to a datetime object
    date_obj = datetime.strptime(date_str, '%m/%Y')
    
    # Get the month name and year
    month_name = date_obj.strftime('%B')
    year = date_obj.year
    
    # Return the formatted month name and year
    return f"{month_name} {year}"
def convert_data(painpoints, months):
    result = []

    # Create the header row
    header_row = [0, 0, "Painpoint"]
    result.append(header_row)
    months = sorted(months, key=custom_sort)

    for i, month in enumerate(months):
        result.append([0, i+1, convert_date(month)])

    for i, (painpoint, data) in enumerate(painpoints.items(), start=1):
        row_data = [i, 0, painpoint]
        result.append(row_data)
        for j, month in enumerate(months):
            block_ids = data.get(month, [])
            result.append([i, j+1, len(block_ids)])
    with open("dana_reviews_output.csv", 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(result)
    return result
# print("cluster reviews")
# print(cluster_reviews(LENS_ID, default_painpoints, -1))
# print("KMEANS for painpoints")
# print(cluster_painpoints_for_topics(LENS_ID, KMEANS))
# print("KMEANS for chunks")
# print(cluster_for_topics(LENS_ID, KMEANS))
# print("DBSCAN")
# print(cluster_for_topics(LENS_ID, DBSCAN))
# print("MOG")
# print(cluster_for_topics(LENS_ID, MIXTURE_OF_GAUSSIANS))
# print("AG")
# print(cluster_for_topics(LENS_ID, AGGLOMERATIVE_CLUSTERING))
