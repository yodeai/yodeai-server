# app/answerer.py
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

from openai import ChatCompletion
llm = OpenAI(temperature=0)

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]



def answer_question(question: str) -> str:
    
    prompt = f"You are answering the questions of freshmen from UC Berkeley. Write a helpful and concise answer for the question ``{question}'' in at most one paragraph."
    response=get_completion(prompt)

    return response