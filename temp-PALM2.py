import time
import vertexai
from vertexai.language_models import TextGenerationModel
from utils import get_completion

def interview(
    temperature: float,
    project_id: str,
    location: str,
) -> str:
    """Ideation example with a Large Language Model"""

    vertexai.init(project=project_id, location=location)
    # TODO developer - override these parameters as needed:
    parameters = {
        "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
        "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }

    model = TextGenerationModel.from_pretrained("text-bison@002")
    response = model.predict(
        "Give me ten interview questions for the role of program manager.",
        **parameters,
    )
    print(f"Response from Model: {response.text}")

    return response.text


if __name__ == "__main__":
    start_time = time.time()
    interview(0.2, "ninth-bonito-399217", "us-west1" )
    #print(get_completion("Give me ten interview questions for the role of program manager.", "gpt-3.5-turbo"))
    end_time = time.time()
    print(end_time-start_time)
