import pandas as pd

happy_words = ['happy', 'joy', 'love', 'excited', 'pleased', 'delighted', 'content', 'cheerful', 'elated', 'thrilled']
sad_words = ['sad','bad', 'angry', 'upset', 'depressed', 'unhappy', 'miserable', 'gloomy', 'downcast', 'sorrowful', 'dejected']

def analyze_sentiment(text):
    words = text.lower().split()
    happy_score = 0
    sad_score = 0

    for i, word in enumerate(words):
        weight = i + 1   
        if word in happy_words:
            happy_score += weight
        if word in sad_words:
            sad_score += weight
    
    print(f"Happy Score: {happy_score}, Sad Score: {sad_score}")
    if happy_score > sad_score:
        return 'Happy'
    elif sad_score > happy_score:
        return 'Sad'
    else:
        return 'Neutral'


if __name__ == "__main__":
    print(analyze_sentiment(input("Enter a sentence: ")))

#----------------------------USING BAYES THEOREM------------------------------------------------
# 
from collections import Counter
import math
import re

happy_words = ['happy','joy','love','excited','pleased','delighted','content','cheerful','elated','thrilled']
sad_words = ['sad','bad','angry','upset','depressed','unhappy','miserable','gloomy','downcast','sorrowful','dejected']

# Training data
training_data = [
    ("i am very happy", "Happy"),
    ("this is a joyful day", "Happy"),
    ("i feel sad and upset", "Sad"),
    ("this is a bad day", "Sad")
]

# Train
happy_counts = Counter()
sad_counts = Counter()
happy_total = sad_total = 0

for text, label in training_data:
    words = re.findall(r'\b\w+\b', text.lower())
    if label == "Happy":
        happy_counts.update(words)
        happy_total += len(words)
    else:
        sad_counts.update(words)
        sad_total += len(words)

vocab = set(happy_counts.keys()).union(sad_counts.keys())

def bayes_sentiment(text):
    words = re.findall(r'\b\w+\b', text.lower())

    log_happy = math.log(0.5)
    log_sad = math.log(0.5)

    for w in words:
        log_happy += math.log((happy_counts[w] + 1) / (happy_total + len(vocab)))
        log_sad += math.log((sad_counts[w] + 1) / (sad_total + len(vocab)))

    return "Happy" if log_happy > log_sad else "Sad"

# Input
print(bayes_sentiment("I am happy now; yesterday I was sad"))
    