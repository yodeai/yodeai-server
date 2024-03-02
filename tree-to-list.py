import ast


unitdistance = 20
inputfilename = "reviews-v3.txt"
inputfile = open(inputfilename, "r")
output = ""

class Node:
    def __init__(self, title, content, children):
        self.title = title
        self.content = content
        self.children = children
        self.count = 0

node_list = []
for _ in range(1000):
    node_list.append(Node("","",[]))

#print((Node3.children)[1].title)

def read_next_node():
    next_node = Node("","",[])
    line = ""
    while True:
        line = inputfile.readline()        
        if not line:
            return False        
        if line.strip():  # Strip removes leading/trailing whitespace
            #print(line)
            break
    print("inside read next node")
    print(line)
    node_id_starts = line.find(":")+1
    node_id = (int)(line[node_id_starts:].strip())
    line = inputfile.readline()        
    title_starts = line.find(":")+1
    if (line[0:title_starts].strip()!="Title:"):
        content = line.strip()
        next_node.title = ""
        next_node.content = content
        next_node.children = []
        node_list[node_id] = next_node
        line = inputfile.readline()
        return node_id

    title = line[title_starts:].strip()
    line = inputfile.readline()        
    line = inputfile.readline() 
    summary_starts = line.find(":")+1
    summary = line[summary_starts:].strip()
    line = inputfile.readline() 
    children_starts = line.find(":")+1
    children_list_str = line[children_starts:].strip()
    #print("children list:")
    #print(children_list_str)
    #print("end children list")
    children_list = ast.literal_eval(children_list_str)

    next_node.title = title
    next_node.content = summary
    next_node.children = children_list
    node_list[node_id] = next_node
    return node_id


def read_nodes():
    root = read_next_node()
    while (read_next_node()):
        continue
    return root

def compute_counts(node):
    if (len(node.children)==0):
        node.count = 1
        return 1
    sum = 0
    for child_id in node.children:
       sum += compute_counts(node_list[child_id])
    node.count = sum
    return sum


def print_node(node, depth):
    global output
    parentmargin = f"style=\"margin-left: {(depth)*unitdistance}px;\""
    childmargin =  f"style=\"margin-left: {(depth+1)*unitdistance}px;\""
    if (node.title == ""):
        output += "<details>\n"
        output += f"<summary {parentmargin}>{node.content} ({node.count})</summary>\n"        
        output += "</details>\n"
        #output += f"<li {parentmargin}>{node.content}</li>\n"        
        return output
    output += "<details>\n"
    output += f"<summary {parentmargin}><b>{node.title}</b> ({node.count})</summary>\n"
    output += f"<div {childmargin}> {node.content} </div>"
    if (len(node.children)>0):
        for child_id in node.children:
            print_node(node_list[child_id], depth+1)
    output += "</details>\n"
    return


root = read_nodes()
compute_counts(node_list[root])
# print("info\n\n")
# print(root)
# for i in range(10):
#     print(node_list[i].title)
#     print(node_list[i].children)
#     print("\n")

print_node(node_list[root], 0)

file = open("tree.html", "w")  # Open file in write mode
file.write(output)
file.close()
