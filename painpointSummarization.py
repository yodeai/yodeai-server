import os
from debug.tools import clearConsole
from utils import getEmbeddings

from extract_msg import Message
from langchain.document_loaders import PyPDFLoader
from docx import Document
from DB import supabaseClient

gpt_model_name = "gpt-3.5-turbo"

# Replace 'your_folder_path' with the path to the folder you want to read files from.
folder_path = '../reviews'

output_file = "reviews.txt"
nonsubstantives_file = "non-substantive-files.txt"

with open(output_file, 'w') as file:
    file.write("First line:\n")
with open(nonsubstantives_file, 'w') as file:
    file.write("First line:\n")

def updateFile(filename, data):
    with open(filename, 'a') as file:
        file.write(data)

data = []
parent = [0]*5000
children = [[] for _ in range(5000)]

debugCluster = 87

rawData = {}


def getPainPoints(review):
    prompt = f"I want to extract the main pain points that the user is facing from the review I provide below. Create a list of these pain points. Each item on the list should focus on a specific pain point that the user mentioned. Start the description of each item on the list with a short expressive name that summarizes the theme of that pain point. Remember, only list pain points, not positive comments from the user.  User Review: ``{review}''"   
    response = get_completion(prompt, gpt_model_name)
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

def getRawData():
    if not (os.path.exists(folder_path) and os.path.isdir(folder_path)):
        print("The specified folder does not exist or is not a directory.")
    else:
        file_path = folder_path+"/"+"reviews.txt"
        with open(file_path, 'r') as file:
            reviews = file.readlines()

    # Combine lines into reviews separated by blank lines
    current_review = ''
    separated_reviews = []
    
    index = 0
    for line in reviews:
        if line.strip():  # Check if the line is not blank
            current_review += line
        else:
            
            if current_review:
                reviewSplits = splitReviewPainpoints(getPainPoints(current_review.strip()))
                for split in reviewSplits:
                    rawData[index] = {"body":split}  
                    index += 1
                current_review = ''

    # Append the last review if the file doesn't end with a blank line
    if current_review:
        reviewSplits = splitReviewPainpoints(getPainPoints(current_review.strip()))
        for split in reviewSplits:
            rawData[index] = {"body":split}  
            index += 1
    return None



def makeDB():
    #clearDB()
    getRawData()

    for index,key in enumerate(rawData):       
        clearConsole(f"makeDB for loop iteration: {index}") 
        text = rawData[key]['body']
        #print(f"text length: {len(text)}")        

        supabaseClient.table('painpoint_summarization').upsert({
                'block_type': "text",
                'content': text,
                'parent_id': 0,  
                'embedding': getEmbeddings(text),
                'reviewer_id': key
        }).execute()



def remakeDB():    
    return

def getFloat(vec):
    return [float(value) for value in vec.split(',')]
    
def loadData():
    data, error = supabaseClient.table('painpoint_summarization').select('block_id', 'content', 'embedding', 'reviewer_id').order('block_id').execute()
    ans = []
    count = 0
    for row in data[1]: 
        print(f"loading row {count} with block_id {row['block_id']}")
        text = row['content']
        # score = 10
        # if (isinstance(row['substantiveness'],int)):
        #     score = row['substantiveness']
        #     print(f"score found for {row['block_id']}")
        # else:
        #     score = getSubstantiveness(text)???
        # update_response, update_error = supabaseClient.table('block_for_lisa')\
        #     .update({'substantiveness': score})\
        #     .eq('block_id', row['block_id'])\
        #     .execute()
        #if (score >= 5):
        ans.append({'content': row['content'], 'embedding': ast.literal_eval(row['embedding']), 'reviewer_id': row['reviewer_id']})
        #count += 1
    return ans



from DB import getBlockIDsOfLens, getBlockTitles, supabaseClient
from debug.tools import clearConsole
from processBlock import processBlock
from utils import get_completion, getEmbeddings
import numpy as np
import ast
import re 
from sklearn.cluster import AffinityPropagation
from celery_tasks.tasks import process_block_task 

def extractFirstInteger(text):
    # Use regular expression to find the first integer in the string
    match = re.search(r'\d+', text)

    if match:
        return int(match.group())
    else:        
        return 0
    
def getSubstantiveness(text):
    if ((not text) or (len(text)<5)):
        return 0
    prompt = f"Does the following text have any concrete substantive information? Assign a score between 0 to 10, with 10 being the most substantive. Answer with a single number, the score alone. Text: ``{text}'' "
    response = get_completion(prompt, gpt_model_name)
    clearConsole( extractFirstInteger(response))
    return extractFirstInteger(response)


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


def getClusterSummaryInfo(cluster, blocksInCluster):
    if (len(blocksInCluster)==0): 
        return {"content": ""}  
    elif (len(blocksInCluster)==1):
        blockID = blocksInCluster[0]['blockID']
        return {"content": data[blockID]['content']}
    
    blocksInCluster = blocksInCluster[0:8]
    text = ""
    for b in blocksInCluster:
        text += data[b['blockID']]['content']
        text += "\n\n"

    prompt = f"""You are summarizing the following text inside qoutes in a concise way.     
    The text contains multiple parts. Each part is about a user pain point. These parts are separated by blank lines. 
    Write a brief summary for the entire text that captures the main theme for all of the given pain points,  preferably no more than 20 words.
    Text: "{text}" Your answer should be formatted as follows: in the first line, write "Main theme:" followed with  your response. """

    # prompt = f"""You are summarizing the following text inside qoutes in a concise way.     
    # The text contains multiple parts, each about a document or an email exchanges between people. These parts are separated by blank lines. 
    # Firstly, find an informative title that reflects the main theme for the entire text. 
    # Then, in a new line write a summary for the entire text preferably no more than 200 words that preserves concrete information such as names that are central to the narratives in the text.
    # Text: "{text}" \n Your answer should be formatted as follows: in the first line, write "Title:" followed with the title, and in the second line write "Summary:" followed with the summary."""
    
    # if (cluster == debugCluster):
    #     print(f"\n\n debugCluster {debugCluster}  promot: {prompt}")
    #     for b in blocksInCluster:
    #         print(f"block in {debugCluster}: {b}\n")
    response = get_completion(prompt, gpt_model_name)
    content_starts = 1+response.find(":")
    content = response[content_starts:].strip()
    return {"content": content}

def add2queue(tail, clusterIndex, summaryInfo):
    #ans.append({'content': row['content'], 'embedding': ast.literal_eval(row['embedding']), 'reviewer_id': row['reviewer_id']})
    if (len(data)<=tail):                
        data.append({'content': "",  'embedding': [], 'reviewer_id': ""})
    data[tail] = {'content': summaryInfo['content'],  'embedding': getEmbeddings(summaryInfo['content']), 'reviewer_id': ""}
    return


def refineTree():
    for index, entry in enumerate(data):
        if len(children[index])==1:            
            print(f"connecting child{children[index][0]} of {index} to its parent {parent[index]}")
            if (parent[index] != 0):
                parent[children[index][0]] = parent[index]
                positionOfIndex = children[parent[index]].index(index)
                children[parent[index]][positionOfIndex] = children[index][0]
            else:
                parent[children[index][0]] = 0
                children[index] = []
            parent[index] = -1
                
def writeNode(node): 
    blockData = ""
    if (len(children[node])>0):
        blockData = f"node: {node}\n Content: {data[node]['content']}\n  children: {children[node]}\n\n"
    else: 
        blockData = f"node: {node}\n Content: {data[node]['content']}\n  Reviewer ID: {data[node]['reviewer_id']}\n\n"
    updateFile(output_file, blockData)
    # for c in childrenInfo:
    #     childID = c['blockID']
    #     supabaseClient.table('result_for_lisa').upsert({"block_id:"}, returning='minimal')
    return

def writeOutput():
    index = len(data) - 1
    while (index >= 0):
        if (parent[index] >= 0):
            writeNode(index)
        index -= 1

def clusterAndSummarize(head, tail):
    clearConsole(f"starting clusterAndSummarize from head:{head} to tail:{tail}")
    affinity_propagation = AffinityPropagation(damping=0.5, preference=None)

    # Fit the model
    embeddings = [row['embedding'] for row in data[head:tail]]

    affinity_propagation.fit(embeddings)

    # Get cluster assignments
    cluster_labels = affinity_propagation.labels_
    clearConsole(cluster_labels)    

    # Find the number of clusters
    n_clusters = len(np.unique(cluster_labels))    
    if (n_clusters == tail-head):
        return False
    print("n_clusters:")
    clearConsole(n_clusters)

    centers = computeCenters(n_clusters, cluster_labels, embeddings)


# compute the similarity between each center and each block in the corresponding cluster
    blocksInCluster = [[] for _ in range(n_clusters)]
    for bindex, clabel in enumerate(cluster_labels):
        blocksInCluster[clabel].append({"blockID": head+bindex , "similarity": np.dot(centers[clabel], embeddings[bindex])})


    # Draw 7 representative articles from each cluster, those closest to the center
    relevantBlocksInCluster = [[] for _ in range(n_clusters)]
    allRelevantBlocks = []
    for c in range(n_clusters):
        ans = getRelevantBlocksInCluster(c, blocksInCluster[c], 100)
        relevantBlocksInCluster[c].extend(ans)
        allRelevantBlocks.extend(ans)

    # generate titles + summaries for each cluster, then recursively summarize the summaries    
    for c in range(n_clusters):
        summaryInfo = getClusterSummaryInfo(c, blocksInCluster[c])
        print(f"cluster:{c}")
        #compute update children and parent, then add the cluster node (parent) to the queue
        children[tail] = [b['blockID'] for b in blocksInCluster[c]]
        #print(f"children of {tail}: {children[tail]}")
        for b in blocksInCluster[c]:
            parent[b['blockID']] = tail            
        add2queue(tail, c, summaryInfo)
        tail = tail+1
    return True   



def clusterData(): 
    # get average embeddings, one for each block in the lens      
    head = 0
    tail = len(data)
    count = 0
    while (head < tail):        
        if (clusterAndSummarize(head, tail) == False):
            break
        head = tail
        tail = len(data)
    return


np.random.seed(42)


#makeDB()
data = loadData()
# for entry in data:
#     print(entry['content'])
#     print(entry['reviewer_id'])
clusterData()
refineTree()
writeOutput()












    
