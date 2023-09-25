def test_answer_question_lens():
    question = "What is the meaning of life?"
    lensID = "2"
    
    #expected_response = "This is a test response from the backend, and the question is: " + question + " and the lensID is: " + lensID
    response = answer_question_lens(question, lensID)
    print(response)
    

def answer_question_lens(question: str, lensID: str) -> str:
    response = "This is a test response from the backend, and the question is: " + question + " and the lensID is: " + lensID
    return response


if __name__ == "__main__":
    test_answer_question_lens()
