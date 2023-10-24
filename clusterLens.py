from DB import getBlockIDsOfLens, getBlockTitles, supabaseClient
from debug.tools import clearConsole
from processBlock import processBlock
from utils import get_completion, getEmbeddings
import numpy as np
import ast
from sklearn.cluster import AffinityPropagation
from celery_tasks.tasks import process_block_task 

def computeCenters(n_clusters, cluster_labels, embeddings):
    #compute cluster centers
    cluster_sizes = np.zeros(n_clusters) 
    cluster_centers =  np.zeros((n_clusters, len(embeddings[0]))) 
    for b in range(len(cluster_labels)):
        cluster_centers[cluster_labels[b]] += np.array(embeddings[b])
        cluster_sizes[cluster_labels[b]] += 1
    for c in range(n_clusters):
        cluster_centers[c] = cluster_centers[c] / cluster_sizes[c]
    return cluster_centers

def getRelevantBlocksInCluster(cluster, blockInfo, topk):
    sorted_list = sorted(blockInfo, key=lambda x: x['similarity'], reverse=True)
    return [sorted_list[i]['blockID'] for i in range(min(topk,len(sorted_list)))]
    
def getPreviewDict(blockIDs):
    ans = {}
    try:
        data, error = supabaseClient.table('block') \
        .select('block_id','preview') \
        .in_('block_id', blockIDs) \
        .execute()
    except Exception as e:
        print(f"Exception occurred while retrieving updated_at, created_at: {e}")
    for entry in data[1]:
        ans[entry['block_id']] = entry['preview']
        if not entry['preview']:
            ans[entry['block_id']] = processBlock(entry['block_id'])
    return ans

def getClusterPreview(blocksInCluster, previewDict):
    previews = ""
    for blockID in blocksInCluster:
        previews += previewDict[blockID] + "\n\n"
    prompt = f"You are generating a one- or two-sentence description  for a set of documents. A summary of each document is given in the following text inside triple qoutes. The summaries are separated by blank lines. The description you generate will be shown to the user as a preview of  the entire set of documents.   Summaries of documents: ```{previews}'''"
    response = get_completion(prompt)
    return response

def getClusterKeywords(blocksInCluster, previewDict):
    previews = ""    
    for blockID in blocksInCluster:
        previews += previewDict[blockID] + "\n\n"
    prompt = f"You are generating  comma-separated keywords  for a set of documents. A summary of each document is given in the following text inside triple qoutes. The summaries are separated by blank lines. The keywords you generate will be shown to the user as a preview of  the entire set of documents. Generate a total of four keywords for the entire set of documents.  Summaries of documents: ```{previews}'''"
    response = get_completion(prompt)
    return response

def clusterLens(lensID): 
    rpc_params = {
        "lensid": lensID
    }
    data, error = supabaseClient.rpc("get_embeddings_of_blocks_in_lens", rpc_params).execute()               
    blockIDs = [((entry['block_id'])) for entry in data[1]]
    
    # get average embeddings, one for each block in the lens
    for entry in data[1]:
        if (not entry['ave_embedding']):
            print("processing block")
            processBlock(entry['block_id'])       
    embeddings = [(ast.literal_eval(entry['ave_embedding'])) for entry in data[1]]
        
        
    # Create and configure the model
    affinity_propagation = AffinityPropagation(damping=0.5, preference=None)

    # Fit the model
    affinity_propagation.fit(embeddings)

    # Get cluster assignments
    cluster_labels = affinity_propagation.labels_
    clearConsole(cluster_labels)    

    # Find the number of clusters
    n_clusters = len(np.unique(cluster_labels))

    # compute cluster centers
    centers = computeCenters(n_clusters, cluster_labels, embeddings)
    
    # compute the similarity between each center and each block in the corresponding cluster
    blocksInCluster = [[] for _ in range(n_clusters)]
    for bindex, clabel in enumerate(cluster_labels):
        blocksInCluster[clabel].append({"blockID": blockIDs[bindex] , "similarity": np.dot(centers[clabel], embeddings[bindex])})
    
    # Draw 7 representative articles from each cluster, those closest to the center
    relevantBlocksInCluster = [[] for _ in range(n_clusters)]
    allRelevantBlocks = []    
    for c in range(n_clusters):
        ans = getRelevantBlocksInCluster(c, blocksInCluster[c], 7)
        relevantBlocksInCluster[c].extend(ans)
        allRelevantBlocks.extend(ans)
        #clearConsole(ans)
        #print(getBlockTitles(ans))

    # printing blocks in each cluster, and its preview
    previewDict = getPreviewDict(allRelevantBlocks)
    for c in range(n_clusters):
        clearConsole(f"cluster: {c}")
        print(relevantBlocksInCluster[c])
        print("\n")
        print(getClusterKeywords(relevantBlocksInCluster[c], previewDict))
        #print(getClusterPreview(relevantBlocksInCluster[c], previewDict))



if __name__ == "__main__":
    np.random.seed(42)
    clusterLens(263)
    # lensID = 6
    # centers = computeCenters(2, [0,1 , 0, 1], [[1,2,3],[10,20,30],[1.1,2.2,3.3],[11,12,13]])
    # print(centers)
    #clusterLens(lensID)
    # blockIDs = getBlockIDsOfLens(lensID)

    # for x in [2, 235, 236, 587, 595]:
    #     clearConsole("starting to process: "+str(x))
    #     processBlock(x)

 
    # for blockID in blockIDs:        
    #     try:
    #         process_block_task.apply_async(args=[blockID])
    #         print("Content processed and chunks stored successfully.")
    #     except Exception as e:
    #         print(f"Exception occurred: {e}")
    

    
