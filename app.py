from flask import Flask
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Global variable to hold the model
model = None

def load_model():
    global model
    if model is None:
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    return model

@app.route("/")  # Make sure to use @app.route and not @app.get
def read_root():
    sentences_1 = ["Japanese food is the best", "this is the new sentence!!"]
    sentences_2 = ["I like ramen", "and I also like sushi.", "and other japanese food."]
    
    # Load the model
    m = load_model()
    
    embeddings_1 = m.encode(sentences_1, normalize_embeddings=True)
    embeddings_2 = m.encode(sentences_2, normalize_embeddings=True)
    similarity = embeddings_1 @ embeddings_2.T
    return {"similarity": similarity.tolist()}

if __name__ == '__main__':
    app.run(debug=True)
