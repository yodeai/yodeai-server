import os
import re
import time
import numpy as np
from debug.tools import clearConsole
from utils import get_completion



reviews = []
def loadReviews():
    reviews.append("Much more accessible for blind users than the web version Up to this point I’ve mostly been using ChatGPT on my windows desktop using Google Chrome. While it’s doable, screen reader navigation is pretty difficult on the desktop site and you really have to be an advanced user to find your way through it. I have submitted numerous feedbacks to open AI about this but nothing has changed on that front. Well, the good news – the iOS app pretty much addresses all of those problems. The UI seems really clean, uncluttered and designed well to be compatible with voiceover, the screen reader built into iOS. I applaud the inclusivity of this design – I only wish they would give the same attention and care to the accessibility experience of the desktop app. I would have given this review five stars but I have just a couple minor quibbles. First, once I submit my prompt, voiceover starts to read aloud ChatGPT‘s response before that response is finished, so I will hear the first few words of the response followed by voiceover reading aloud the “stop generating” button, which isn’t super helpful. It would be great if you could better coordinate this alert so that it didn’t start reading the message until it had been fully generated. The other thing I’d like is a Feedback button easily accessible from within the main screen of the app, to make it as easy as possible to get continuing suggestions and feedback from your users. Otherwise, fantastic app so far!")
    reviews.append("Much anticipated, wasn’t let down. I’ve been a user since it’s initial roll out and have been waiting for a mobile application ever since using the web app. For reference I’m a software engineering student while working in IT full time. I have to say GPT is an crucial tool. It takes far less time to get information quickly that you’d otherwise have to source from stack-overflow, various red-hat articles, Ubuntu articles, searching through software documentation, Microsoft documentation ect. Typically chat gpt can find the answer in a fraction of a second that google can. Obviously it is wrong, a lot. But to have the ability to get quick information on my phone like I can in the web browser I’m super excited about and have already been using the mobile app since download constantly. And I’m excited for the future of this program becoming more accurate and it seems to be getting more and more precise with every roll out. Gone are the days scouring the internet for obscure pieces of information, chat gpt can find it for you with 2 or 3 prompts. I love this app and I’m so happy it’s on mobile now. The UI is also very sleek, easy to use. My only complaint with the interface is the history tab at the top right. I actually prefer the conversation tabs on the left in the web app but I understand it would make the app kind of clunky especially on mobile since the screen size is smaller. Anyway, awesome app 5 stars.")
    reviews.append("Great potential, overwhelmed system I will start by saying that when Chat GPT works, it is incredible. I started with the free version and had a positive experience except that pieces of conversations appeared to disappear. At one point, I lost a one hour conversation which I had to go back and repeat, as I was trying to plan something and hadn’t written things down. I chalked that up to a glitch and decided to upgrade to the paid version. When I first started with the new version, it seemed to misunderstand lots of words, which hadn’t been a problem using the free version. Since I’ve upgraded, about 75% of the time the system either has not been available, experiences an unknown error, or experiences heavy volume, making my purchase useless. As an example, I’ve tried several times today and have not been able to use it most of the day. The one time I did use it, if was skipping words and repeating the same sentences over with slightly different wording. I’m not sure what is going on, but if a system is not available, it is not worth subscribing. I love the technology and think it has tremendous potential, so I hope these links in the system can be worked out soon or I will be cancelling and looking at other options.")
    
def getPainPoints(review):
    prompt = f"I want to extract the main pain points that the user is facing from the review I provide below. Create a list of these pain points. Each item on the list should focus on a specific pain point that the user mentioned. Start the description of each item on the list with a short expressive name that summarizes the theme of that pain point. Remember, only list pain points, not positive comments from the user.  User Review: ``{review}''"   
    response = get_completion(prompt, "gpt-3.5-turbo")
    clearConsole(response)
    return
    # title_starts = 1+response.find(":")
    # title_ends = response.find("\n")    
    # title = response[title_starts:title_ends].strip()
    # all_but_title = response[title_ends:]
    # summary_starts = 1+all_but_title.find(":")
    # summary = all_but_title[summary_starts:].strip()
    # if (len(text)<1000):
    #     summary = text
    # clearConsole(response)
    # return {"title": title, "summary": summary}

def parseResult(input_string):
    # Splitting the string using "\n" and "."
    result = re.split(r'[\n.]', input_string)

    # Removing empty strings from the result
    result = [part.strip() for part in result if (len(part.strip())>3)]

    splits=[]
    for i, part in enumerate(result, 1):
        splits.append(part)
    return splits

np.random.seed(42)


#loadReviews()
#getPainPoints(reviews[0])

start_time = time.time()
print( get_completion("who won the worldcup in 2018?"))
end_time = time.time()
print(end_time-start_time)

#makeDB()
#data = loadData()
#clusterData()
#refineTree()
#writeOutput()












    
