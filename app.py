from flask import Flask
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

@app.get("/")
def read_root():
    sentences_1 = ["Japanese food is the best", "this is the new sentence!!"]
    sentences_2 = ["I like ramen", "and I also like sushi.", "and other japanese food."]
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
    embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
    similarity = embeddings_1 @ embeddings_2.T
    return {"similarity": similarity.tolist()}
