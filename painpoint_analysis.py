from DB import supabaseClient
from sklearn.cluster import KMeans, DBSCAN as db
import json
from utils import getEmbeddings, get_completion
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "gpt-3.5-turbo"
NUM_CLUSTERS = 5
LENS_ID = 959
KMEANS = 'KMEANS'
# DBSCAN = 'DBSCAN'
# MIXTURE_OF_GAUSSIANS = "MOG"
# AGGLOMERATIVE_CLUSTERING = "AG"
default_painpoints = ['Inconsistent and limited AI responses.', 'System crashes and unavailability.', 'Login loop and server crashing.', 'Limited features in paid version.', 'Incomplete message generation.']

def get_block_ids(lens_id):
    block_ids, count = supabaseClient.table('lens_blocks').select("block_id").eq('lens_id', lens_id).execute()
    return block_ids[1]

def get_chunks_from_block_ids(block_ids):
    block_ids_list = [block_info['block_id'] for block_info in block_ids]
    chunks, count = supabaseClient.table('chunk').select('content', 'embedding', 'block_id').in_('block_id', block_ids_list).execute()
    # return a mapping of the embedding to the content
    mapping = {}
    embeddings = []
    for chunk in chunks[1]:
        embedding = json.loads(chunk['embedding'])
        mapping[chunk['embedding']] = chunk
        embeddings.append(embedding)
    return mapping, embeddings

def cosine_similarity_vectors(vec1, vec2):
    # Ensure the vectors are not empty
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    # Calculate cosine similarity
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    return similarity

def find_closest_5_chunks(cluster_centroid, chunk_embeddings):
    # Calculate cosine similarity between the cluster centroid and each chunk embedding
    similarities = [cosine_similarity_vectors(cluster_centroid, json.loads(emb["embedding"])) for emb in chunk_embeddings]
    
    # Sort chunks based on similarity scores (in descending order)
    sorted_chunks = sorted(zip(chunk_embeddings, similarities), key=lambda x: x[1], reverse=True)
    
    # Select top 5 chunks
    closest_chunks = sorted_chunks[:5]
    
    return closest_chunks

def cluster_for_topics(lens_id, method=KMEANS):
    block_ids = get_block_ids(lens_id)
    chunks_mapping, chunks_list = get_chunks_from_block_ids(block_ids)
    chunks_list_str = list(chunks_mapping.keys())
    if method == KMEANS:
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
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
            prompt = f"Please output one main painpoint from this content, which is a collection of user reviews on a product: {text}. OUTPUT THE PAINPOINT IN 4-5 WORDS ONLY."
            topic = get_completion(prompt, MODEL_NAME)
            topics.append(topic)
        return topics

def cluster_reviews(lens_id, painpoints, method=KMEANS):
    painpoint_embeddings = [getEmbeddings(painpoint) for painpoint in painpoints]
    block_ids = get_block_ids(lens_id)
    chunks_mapping, chunks_list = get_chunks_from_block_ids(block_ids)
    chunks_list_str = list(chunks_mapping.keys())
        
    if method == 'KMEANS':
        kmeans = KMeans(n_clusters=len(painpoints), init=painpoint_embeddings, n_init=len(painpoint_embeddings))
        # Get cluster assignments for each chunk
        kmeans.fit_predict(chunks_list)
        cluster_assignments = kmeans.labels_
        # Collect chunks belonging to each cluster
        cluster_chunks = {}
        for i, assignment in enumerate(cluster_assignments):
            if assignment not in cluster_chunks:
                cluster_chunks[assignment] = []
            emb = chunks_list_str[i]
            cluster_chunks[assignment].append(chunks_mapping[emb])
        painpoint_to_block_id = {}
        for cluster, chunks in cluster_chunks.items():
            # Choose the representative pain point for the cluster
            # In this example, we simply choose the first pain point for simplicity
            representative_painpoint = painpoints[cluster]
            block_ids = set([chunk['block_id'] for chunk in chunks])
            
            # Associate the cluster with the representative pain point
            painpoint_to_block_id[representative_painpoint] = block_ids
        return painpoint_to_block_id

# print("cluster reviews")
# print(cluster_reviews(LENS_ID, default_painpoints))
# print("KMEANS")
# print(cluster_for_topics(LENS_ID, KMEANS))
# print("DBSCAN")
# print(cluster_for_topics(LENS_ID, DBSCAN))
# print("MOG")
# print(cluster_for_topics(LENS_ID, MIXTURE_OF_GAUSSIANS))
# print("AG")
# print(cluster_for_topics(LENS_ID, AGGLOMERATIVE_CLUSTERING))