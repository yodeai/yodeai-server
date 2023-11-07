#parms: 
# timeout, 
# getRelevantBlocksInCluster: takes input
# simply_summarize: takes input
# getSummaryData4RawDataBlock: inside the function 
import limits
import os
from debug.tools import clearConsole
from utils import get_title, getEmbeddings, remove_invalid_surrogates, simply_summarize

from extract_msg import Message
from langchain.document_loaders import PyPDFLoader
from docx import Document
from DB import supabaseClient

# Replace 'your_folder_path' with the path to the folder you want to read files from.
folder_path = '../for-Shadi/articles'

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

    text = remove_invalid_surrogates(msg.body)
    return text

def processPDF(file_name):
    loader = PyPDFLoader(folder_path+"/"+file_name)    
    pages = []
    pages.extend(loader.load())
    content = ""
    for page in pages:
        content = content + page.page_content
    text = remove_invalid_surrogates(content)        
    return text






import os
import pytesseract
from PIL import Image
#pytesseract.pytesseract.tesseract_cmd = '/Users/afshinni/Dropbox/github/yodeai-server/venv/bin/pytesseract'

def extractIntFromFileName(filename):
    return int(filename.split('.')[0])

def processImageFolder(inner_folder):
    clearConsole(f"innder_folder:{inner_folder}")
    inner_folder_path = folder_path+'/'+inner_folder
    output_pdf = folder_path+'/processed_'+inner_folder+'.pdf'

    file_name_list = []
    for file_name in os.listdir(inner_folder_path):
        if (file_name[0] != '.'):
            file_name_list.append(file_name)
    sorted_file_name_list =  sorted(file_name_list, key=extractIntFromFileName)

    text = ""
    for filename in sorted_file_name_list: 
        if filename.endswith(".jpg"):
            file_path = os.path.join(inner_folder_path, filename)
            # Open the image using PIL (Python Imaging Library)
            image = Image.open(file_path)

            # Use pytesseract to perform OCR on the image and extract text
            text += pytesseract.image_to_string(image)
    text = remove_invalid_surrogates(text)
    return text
        
# import fitz 
# def processPDFwithTextAsImage(file_name):
#     pdf_path = folder_path+'/'+file_name
#     text = ""
#     try:
#         pdf_document = fitz.open(pdf_path)
#         for page_num in range(pdf_document.page_count):
#             page = pdf_document.load_page(page_num)
#             text += page.get_text()
#             clearConsole(page.get_text())
#     except Exception as e:
#         print(f"Error: {str(e)}")     
#     return

def processDOCX(file_name):
    doc = Document(folder_path+"/"+file_name)

    # Iterate through paragraphs in the document
    text = ""
    for paragraph in doc.paragraphs:
        text += (paragraph.text + "\n")
    text = remove_invalid_surrogates(text)
    return text

def getRawData():
    if not (os.path.exists(folder_path) and os.path.isdir(folder_path)):
        print("The specified folder does not exist or is not a directory.")
    else:
        # List all files in the specified folder.
        count = 0
        file_names = os.listdir(folder_path)        
        for name in file_names:
            path = os.path.join(folder_path, name)
            count += 1
            if (name[0] == '.'):
                continue
            if  (os.path.isdir(path)):
                rawData[name] = {"body": processImageFolder(name)}
                continue             
            file_type = name.split('.')[-1] 
            if (file_type.lower() == "msg"):
                rawData[name] = {"body": processMSG(name)}
            elif (file_type.lower() == "pdf"):
                rawData[name] = {"body": processPDF(name)}
            elif (file_type.lower() == "docx"):  
                rawData[name] = {"body": processDOCX(name)}
        print(f"len of rawdata: {len(rawData)}")



def getSummaryData4RawDataBlock(text):
    if (len(text) > 5000): #on average the token limit is around 20000 words        
        summary = simply_summarize(text)
        title = get_title(summary)
        return {"title": title, "summary": summary} 
    prompt =  f"You are summarizing text involving documents in a concise way. The summary should try to preserve concrete information that are central to the narrative of the text. Firstly, give the following text an informative title that reflects the main theme of the text.  Then, in a new line write a summary preferably no more than 200 words for the following text: ```{text}''' Your answer should be formatted as follows: in the first line, write ``Title:'' followed with the title , and  in the second line write  ``Summary:'' followed with the summary."
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
    #clearConsole(response)
    return {"title": title, "summary": summary}


def clearDB():
    response, error = supabaseClient.table('block_for_shadi').select("*").execute()    
    allRows = response[1]
    for row in allRows:        
        supabaseClient.table('block_for_shadi').delete().eq("block_id", row["block_id"]).execute()


def makeDB():
    clearDB()
    clearConsole("begin: getrawdata")
    getRawData()
    clearConsole(len(rawData))
    for index,key in enumerate(rawData):       
        clearConsole(f"makeDB for loop iteration: {index}") 
        text = rawData[key]['body']
        print(f"text length: {len(text)}")
    
        summaryInfo = getSummaryData4RawDataBlock(text)
        print("finished get_completion")
        if (len(rawData[key]['body'])<800):
            summaryInfo['summary'] = rawData[key]['body']

        #clearConsole("block data:")
        #print(f"content: {rawData[key]['body']}")
        print(f"title': {summaryInfo['title']}")
        print(f"summary': {summaryInfo['summary']}")
        print(f"file_name: {key}")
        
        supabaseClient.table('block_for_shadi').upsert({
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
    data, error = supabaseClient.table('block_for_shadi').select('block_id',  'summary', 'title', 'embedding','file_name', 'substantiveness').order('block_id').execute()
    ans = []
    count = 0
    for row in data[1]: 
        print(f"loading row {count} with block_id {row['block_id']}")
        text = row['summary']
        score = 10
        # if (isinstance(row['substantiveness'],int)):
        #     score = row['substantiveness']
        #     print(f"score found for {row['block_id']}")
        # else:
        #     score = getSubstantiveness(text)
        update_response, update_error = supabaseClient.table('block_for_shadi')\
            .update({'substantiveness': score})\
            .eq('block_id', row['block_id'])\
            .execute()
        if (score >= 5):
            ans.append({'title': row['title'], 'summary': row['summary'], 'embedding': ast.literal_eval(row['embedding']), 'file_name': row['file_name']})
        count += 1
    return ans

def addEmbeddings():
    data, error = supabaseClient.table('block_for_shadi').select('block_id', 'summary', 'embedding').order('block_id').execute()
    ans = data[1]
    for index,element in enumerate(data[1]):
        print(f"getting embedding for {index}")
        element['embedding'] = getEmbeddings(element['summary'])
    supabaseClient.table('block_for_shadi').upsert(ans, returning='minimal')
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

#returns  summary

def getSummary4Blocks(text):
    prompt = f"""You are summarizing the following text inside qoutes in a concise way.     
    The text contains multiple parts, each about a document. Each parts is separated from the next part by two blank lines. 
    Write a summary for the entire text in at most 200 to 300 words that preserves concrete information that are central to the narratives in the text.
    Text: "{text}" """
    
    response = get_completion(prompt)    
    return response     

#returns title and summary; summary computed after splitting to smaller groups 
def getClusterSummaryInfo(cluster, blocksInCluster):
    if (len(blocksInCluster)==0): 
        return {"title": "", "summary": ""}  
    elif (len(blocksInCluster)==1):
        blockID = blocksInCluster[0]['blockID']
        return {"title": data[blockID]['title'], "summary": data[blockID]['summary']}
    
    relevantIndex = min(len(blocksInCluster), 10)
    blocksInCluster = blocksInCluster[0:relevantIndex]

    pieces = []
    text = ""
    for index, block in enumerate(blocksInCluster):
        newtext = data[block['blockID']]['summary']
        if (len(text+newtext)>limits.maxChars4Prompt):
            pieces.append(text)
            text = newtext+"\n\n\n"
        else:
            text += newtext
            text += "\n\n\n"
    if (len(text)>0):
        pieces.append(text)
    
    summaries = ""
    for text in pieces:
        summaries += (getSummary4Blocks(text)+"\n\n\n")


    prompt = f"""You are summarizing the following text inside qoutes in a concise way.     
    The text contains multiple parts, each about a document. Each parts is separated from the next part by two blank lines. 
    Firstly, find an informative title that reflects the main theme for the entire text. 
    Then, in a new line write a summary for the entire text in at most 200 to 300 words that preserves concrete information that are central to the narratives in the text.
    Text: "{summaries}" \n Your answer should be formatted as follows: in the first line, write "Title:" followed with the title, and in the second line write "Summary:" followed with the summary."""
    
    print(f"getting cluster summary for cluster {cluster}")
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

def RecursivelySummarize():
    clusterData()
    refineTree()
    writeOutput()

def partitionBasedOnThemes(themes):
    themeEmbeddings = [getEmbeddings(theme) for theme in themes]
    clusterLabel = [0] * len(data)
    for entryIndex, entry in enumerate(data):
        clearConsole(f"for loop active at {entryIndex}")
        maxSimilarity = 0
        maxThemeIndex = 0
        for themeIndex, themeEmbedding in enumerate(themeEmbeddings):
            similarity = np.dot(entry['embedding'], themeEmbedding)
            print(f"themeIndex:{themeIndex}, similarity: {similarity}")
            if (similarity > maxSimilarity):
                maxSimilarity = similarity
                maxThemeIndex = themeIndex
        clusterLabel[entryIndex] = maxThemeIndex 
    return clusterLabel    
    #ans.append({'title': row['title'], 'summary': row['summary'], 'embedding': ast.literal_eval(row['embedding']), 'file_name': row['file_name']})

def assignTexttoTheme(text, themes):
    strThemes = str(themes)
    strThemes = strThemes[1:len(strThemes)-1]    
    prompt =  f"I am giving you a list of themes that are separated by commas: ``{themes}''. You are deciding to which of these themes the following text fits the best: ``{text}''  Your answer should be formatted as follows: in the first line, write ``Theme:'' followed with the theme that you choose from the list of themes I gave you."
    return get_completion(prompt)

np.random.seed(50)
#makeDB()
data = loadData()
RecursivelySummarize()
