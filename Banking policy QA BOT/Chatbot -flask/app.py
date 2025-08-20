import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("/mnt/data/Dataset_Banking_chatbot.csv", encoding="latin1")

# Prepare data
queries = df["Query"].tolist()
responses = df["Response"].tolist()

# Vectorize queries
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(queries)

def chatbot(user_input):
    # Transform user input
    user_vec = vectorizer.transform([user_input])
    
    # Compute cosine similarity
    similarities = cosine_similarity(user_vec, X)
    idx = similarities.argmax()
    
    return responses[idx]

# Example conversation
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye!")
        break
    answer = chatbot(user_input)
    print("Chatbot:", answer)
