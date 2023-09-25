# app/answerer.py

from langchain.llms import OpenAI

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