import os
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def demo():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
