import os
from fastapi import FastAPI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

app = FastAPI()

@app.get("/")
def demo():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Compute embeddings for a list of sentences
    sentences = ["This is a sentence.", "This is another sentence."]
    embeddings = hf_embeddings.embed_documents(sentences)

    # Store the length of embeddings
    lengths = []
    for i, embedding in enumerate(embeddings):
        print(f"Sentence {i+1} embedding: {embedding}")
        lengths.append(len(embedding))

    return {"embedding_lengths": lengths}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
