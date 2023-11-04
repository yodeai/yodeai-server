import os
from debug.tools import clearConsole
from utils import getEmbeddings

from extract_msg import Message
from langchain.document_loaders import PyPDFLoader
from docx import Document
from DB import supabaseClient

# Replace 'your_folder_path' with the path to the folder you want to read files from.
folder_path = '../for-Lisa/emails'

output_file = "result.txt"
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
def processMSG(file_name):
    try:
        msg = Message(folder_path+"/"+file_name)
    except Exception as e:
        print("Error reading msg file ", e)

    text = msg.body
    rawData[file_name]={"body":text}
    return

def processPDF(file_name):
    loader = PyPDFLoader(folder_path+"/"+file_name)    
    pages = []
    pages.extend(loader.load())
    content = ""
    for page in pages:
        content = content + page.page_content
    rawData[file_name]={"body":content}        
    return

def processDOCX(file_name):
    doc = Document(folder_path+"/"+file_name)

    # Iterate through paragraphs in the document
    text = ""
    for paragraph in doc.paragraphs:
        text += (paragraph.text + "\n")
    rawData[file_name]={"body":text}
    return

def getRawData():
    if not (os.path.exists(folder_path) and os.path.isdir(folder_path)):
        print("The specified folder does not exist or is not a directory.")
    else:
        # List all files in the specified folder.
        count = 0
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            count += 1
            clearConsole(file_name)
            print(count)
            file_type = file_name.split('.')[-1] 
            if (file_type == "msg"):
                processMSG(file_name)
            elif (file_type == "pdf"):
                processPDF(file_name)
            elif (file_type == "docx"):  
                processDOCX(file_name)
            else:
                continue





def getSummaryData4RawDataBlock(text):
    prompt =  f"You are summarizing text involving documents or email communications in a concise way. The summary should try to preserve concrete information that are central to the narrative of the text, possibly such as names. Firstly, give the following text an informative title that reflects the main theme of the text.  Then, in a new line write a summary preferably no more than 200 words for the following text: ```{text}''' Your answer should be formatted as follows: in the first line, write ``Title:'' followed with the title , and  in the second line write  ``Summary:'' followed with the summary."
    #clearConsole(prompt)
    response = get_completion(prompt)
    title_starts = 1+response.find(":")
    title_ends = response.find("\n")    
    title = response[title_starts:title_ends].strip()
    all_but_title = response[title_ends:]
    summary_starts = 1+all_but_title.find(":")
    summary = all_but_title[summary_starts:].strip()
    if (len(text)<1000):
        summary = text
    clearConsole(response)
    return {"title": title, "summary": summary}


def clearDB():
    response, error = supabaseClient.table('block_for_lisa').select("*").execute()    
    allRows = response[1]
    for row in allRows:        
        supabaseClient.table('block_for_lisa').delete().eq("block_id", row["block_id"]).execute()

def makeDB():
    #clearDB()
    getRawData()

    for index,key in enumerate(rawData):       
        clearConsole(f"makeDB for loop iteration: {index}") 
        text = rawData[key]['body']
        print(f"text length: {len(text)}")        
        summaryInfo = getSummaryData4RawDataBlock(text)
        print("finished get_completion")
        if (len(rawData[key]['body'])<800):
            summaryInfo['summary'] = rawData[key]['body']

        supabaseClient.table('block_for_lisa').upsert({
                'block_type': "text",
                'content': rawData[key]['body'],
                'parent_id': 0,
                'title': summaryInfo['title'],  
                'summary': summaryInfo['summary'],  
                'embedding': getEmbeddings(summaryInfo['summary']),
                'file_name': key
        }).execute()



def remakeDB():    
    return

def getFloat(vec):
    return [float(value) for value in vec.split(',')]
    
def loadData():
    data, error = supabaseClient.table('block_for_lisa').select('block_id',  'summary', 'title', 'embedding','file_name', 'substantiveness').order('block_id').execute()
    ans = []
    count = 0
    for row in data[1]: 
        print(f"loading row {count} with block_id {row['block_id']}")
        text = row['summary']
        score = 10
        if (isinstance(row['substantiveness'],int)):
            score = row['substantiveness']
            print(f"score found for {row['block_id']}")
        else:
            score = getSubstantiveness(text)
        update_response, update_error = supabaseClient.table('block_for_lisa')\
            .update({'substantiveness': score})\
            .eq('block_id', row['block_id'])\
            .execute()
        if (score >= 5):
            ans.append({'title': row['title'], 'summary': row['summary'], 'embedding': ast.literal_eval(row['embedding']), 'file_name': row['file_name']})
        count += 1
    return ans

def addEmbeddings():
    data, error = supabaseClient.table('block_for_lisa').select('block_id', 'summary', 'embedding').order('block_id').execute()
    ans = data[1]
    for index,element in enumerate(data[1]):
        print(f"getting embedding for {index}")
        element['embedding'] = getEmbeddings(element['summary'])
    supabaseClient.table('block_for_lisa').upsert(ans, returning='minimal')
    return

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
    response = get_completion(prompt)
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
        return {"title": "", "summary": ""}  
    elif (len(blocksInCluster)==1):
        blockID = blocksInCluster[0]['blockID']
        return {"title": data[blockID]['title'], "summary": data[blockID]['summary']}
    
    blocksInCluster = blocksInCluster[0:8]
    text = ""
    for b in blocksInCluster:
        text += data[b['blockID']]['summary']
        text += "\n\n"

    prompt = f"""You are summarizing the following text inside qoutes in a concise way.     
    The text contains multiple parts, each about a document or an email exchanges between people. These parts are separated by blank lines. 
    Firstly, find an informative title that reflects the main theme for the entire text. 
    Then, in a new line write a summary for the entire text preferably no more than 200 words that preserves concrete information such as names that are central to the narratives in the text.
    Text: "{text}" \n Your answer should be formatted as follows: in the first line, write "Title:" followed with the title, and in the second line write "Summary:" followed with the summary."""
    
    # if (cluster == debugCluster):
    #     print(f"\n\n debugCluster {debugCluster}  promot: {prompt}")
    #     for b in blocksInCluster:
    #         print(f"block in {debugCluster}: {b}\n")
    response = get_completion(prompt)
    title_starts = 1+response.find(":")
    title_ends = response.find("\n")    
    title = response[title_starts:title_ends].strip()
    all_but_title = response[title_ends:]
    summary_starts = 1+all_but_title.find(":")
    summary = all_but_title[summary_starts:].strip()
    return {"title": title, "summary": summary}

def add2queue(tail, clusterIndex, summaryInfo):
    if (len(data)<=tail):                
        data.append({'title': "", 'summary': "", 'embedding': [], 'file_name': ""})
    data[tail] = {'title': summaryInfo['title'], 'summary': summaryInfo['summary'], 'embedding': getEmbeddings(summaryInfo['summary']), 'file_name': ""}
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
        blockData = f"node: {node}\n Title: {data[node]['title']}\n Summary: {data[node]['summary']}\n children: {children[node]}\n\n"
    else: 
        blockData = f"node: {node}\n Title: {data[node]['title']}\n Summary: {data[node]['summary']}\n file name: {data[node]['file_name']}\n\n"
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
clusterData()
refineTree()
writeOutput()












    
