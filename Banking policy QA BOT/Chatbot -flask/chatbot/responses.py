import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("C:/Users/DELL/Desktop/Banking policy QA BOT/data/archive/Dataset_Banking_chatbot.csv", encoding="latin1")

# Prepare data
queries = df["Query"].tolist()
responses = df["Response"].tolist()

# Vectorize queries
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(queries)

def get_response(user_input: str) -> str:
    """Return the most relevant response for a given user input."""
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X)
    idx = similarities.argmax()
    return responses[idx]

# Example
print(get_response("How can I open an account?"))
print(get_response("Tell me how to check my balance"))
