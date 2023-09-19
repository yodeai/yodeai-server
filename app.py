import os
import requests
import json
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

@app.get("/")
def demo():
    # Set up the request headers and data
    headers = {"Authorization": os.environ.get("HUGGINGFACEHUB_API_KEY"), "Content-Type": "application/json"}
    data = {"inputs": ["This is a sentence.", "This is another sentence.", "this is a sentence about Japanese food", "Sushi is nice"]}

    # Send the request to the Hugging Face API
    response = requests.post("https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5", headers=headers, data=json.dumps(data))
    #print(response.content)

    # Extract the embeddings from the response
    embeddings = json.loads(response.content.decode("utf-8"))

    # Calculate the pairwise similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Print the similarity matrix
    for i, row in enumerate(similarity_matrix):
        print(f"Sentence {i+1} similarity: {row}")

    return {"similarity_matrix": similarity_matrix.tolist()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)