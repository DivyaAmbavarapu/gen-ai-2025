# utils.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BankingChatbot:
    def __init__(self, dataset_path: str, encoding: str = "latin1"):
        """
        Initialize the chatbot with dataset.
        """
        # Load dataset safely
        df = pd.read_csv("data/archive/Dataset_Banking_chatbot.csv", encoding=encoding)
        df.columns = df.columns.str.strip()   # Remove any spaces in column names
        
        if "Query" not in df.columns or "Response" not in df.columns:
            raise ValueError("Dataset must contain 'Query' and 'Response' columns.")
        
        # Store queries and responses
        self.queries = df["Query"].astype(str).tolist()
        self.responses = df["Response"].astype(str).tolist()
        
        # Vectorize queries
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.queries)

    def get_response(self, user_input: str, threshold: float = 0.2) -> str:
        """
        Return the most relevant response for a given user input.
        If similarity is below threshold, return fallback response.
        """
        if not user_input.strip():
            return "Please enter a valid question."
        
        user_vec = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vec, self.X)
        
        # Safeguard
        if similarities.size == 0:
            return "Sorry, I couldn't process your question."
        
        score = similarities.max()
        idx = similarities.argmax()
        
        if score < threshold:
            return "Sorry, I didn't quite understand that. Could you rephrase?"
        
        return self.responses[idx]

