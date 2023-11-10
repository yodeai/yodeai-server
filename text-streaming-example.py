import time
# the right choice for deployement would be the commented import below
# however, this import doesn't work! its list of models is empty, 
# and it doesn't know text-bison or text-bison-001 
# from vertexai.language_models import TextGenerationModel
from vertexai.preview.language_models import TextGenerationModel


t0 = time.time()
text_model = TextGenerationModel.from_pretrained("text-bison@001")
t1 = time.time()


#response = text_model.predict(prompt=prompt, temperature=0.2)

parameters = {
    "temperature": 0.2,  # Temperature controls the degree of randomness in token selection.
    "max_output_tokens": 300,  # Token limit determines the maximum amount of text output.
    "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
    "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
}  
prompt="What are the main ways for getting kidney transplantation in the US?"
responses = text_model.predict_streaming(prompt, **parameters)
t2 = time.time()
text = ""
for response in responses:
    text += response.text
    
print(t1-t0)    
print(t2-t1)
print(text)

