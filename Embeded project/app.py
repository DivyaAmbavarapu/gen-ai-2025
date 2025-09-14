from bs4 import BeautifulSoup
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import numpy as np

# A class to perform text classification using Hugging Face and XGBoost
class TextClassifier:
    """
    This class handles the entire workflow of a text classification task:
    1. Preprocessing data from a combined text file.
    2. Generating text embeddings using a Hugging Face model.
    3. Training and evaluating an XGBoost classifier.
    4. Making a prediction on new, unseen data.
    """
    def __init__(self, model_name="distilbert-base-uncased"):
        """
        Initializes the TextClassifier with a specified Hugging Face model.

        Args:
            model_name (str): The name of the Hugging Face model to use for embeddings.
                              Default is 'distilbert-base-uncased'.
        """
        print("Loading Hugging Face tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = None
        self.label_mapping = {}
        self.reverse_label_mapping = {}

    def preprocess_data(self, file_path):
        """
        Reads a combined text file containing multiple HTML snippets, extracts
        the full text and the last two words for each, and creates a DataFrame.

        Args:
            file_path (str): The path to the combined text file, or the file content
                             as a string.

        Returns:
            pd.DataFrame: A DataFrame with 'x_text' (full text) and 'y_label'
                          (last two words) columns.
        """
        print(f"Reading and preprocessing data from '{file_path}'...")
        all_x_text = []
        all_y_labels = []

        full_content = ""
        # Check if the input is a file path or a string with the content
        if isinstance(file_path, str) and "--- Start of file:" in file_path:
            full_content = file_path
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    full_content = f.read()
            except FileNotFoundError:
                print(f"Error: The file '{file_path}' was not found.")
                return pd.DataFrame()

        # Split the content by the file start markers to get individual tickets
        ticket_blocks = re.split(r'--- Start of file: .*/n/n', full_content)

        # Skip the first empty element from the split
        for block in ticket_blocks[1:]:
            # Use BeautifulSoup to get the plain text from the HTML block
            soup = BeautifulSoup(block, 'html.parser')
            text_content = soup.get_text(separator=' ', strip=True)
            
            # The 'x' variable is the full text
            x_text = text_content
            
            # The 'y' variable is the last two words of the text
            # We use a simple split and a check to handle short sentences
            words = text_content.split()
            if len(words) >= 2:
                y_label = ' '.join(words[-2:])
            else:
                y_label = ' '.join(words)
            
            all_x_text.append(x_text)
            all_y_labels.append(y_label)

        df = pd.DataFrame({
            'x_text': all_x_text,
            'y_label': all_y_labels
        })
        
        print("/nPreprocessed DataFrame:")
        print(df)
        
        return df

    def create_embeddings(self, texts):
        """
        Generates text embeddings for a list of texts using the Hugging Face model.

        Args:
            texts (list): A list of text strings.

        Returns:
            torch.Tensor: A tensor containing the text embeddings.
        """
        print("Generating text embeddings...")
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def train_and_evaluate(self, df):
        """
        Trains an XGBoost classifier and evaluates its performance on the dataset.

        Args:
            df (pd.DataFrame): The preprocessed DataFrame.
        """
        if df.empty:
            print("No data to train or evaluate.")
            return

        # Create numerical labels for the target variable
        unique_labels = df['y_label'].unique()
        self.label_mapping = {label: i for i, label in enumerate(unique_labels)}
        self.reverse_label_mapping = {i: label for label, i in self.label_mapping.items()}
        
        y_labels = df['y_label'].map(self.label_mapping)

        # Generate embeddings for the text data
        embeddings = self.create_embeddings(df['x_text'].tolist())

        # For a small, single-class dataset, we skip the train/test split
        # as it is not statistically meaningful and can cause errors.
        X_train = embeddings.numpy()
        y_train = y_labels.tolist()

        # Initialize and train the XGBoost classifier
        print("/nTraining XGBoost model...")
        self.classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.classifier.fit(X_train, y_train)
        
        # Since all data points belong to the same class, evaluation will show
        # perfect scores. We can demonstrate prediction on one of the samples.
        print("/n--- Model Evaluation ---")
        y_pred = self.classifier.predict(X_train)
        
        print(f"Accuracy: {accuracy_score(y_train, y_pred):.2f}")
        print("/nClassification Report:")
        print(classification_report(y_train, y_pred, target_names=self.label_mapping.keys()))

    def predict_new_ticket(self, new_text):
        """
        Makes a prediction on a new, unseen text sample.

        Args:
            new_text (str): The text of the new ticket to classify.
        """
        if self.classifier is None:
            print("Error: The model has not been trained yet.")
            return

        print("/n--- Predicting New Ticket ---")
        new_embedding = self.create_embeddings([new_text]).numpy()
        prediction = self.classifier.predict(new_embedding)
        predicted_label = self.reverse_label_mapping[prediction[0]]
        
        print(f"New Ticket Text: '{new_text}'")
        print(f"Predicted Category: {predicted_label}")


# Main execution block
if __name__ == "__main__":
    # The path to your combined HTML content file
    combined_file_path = 'C:/Users/DELL/Desktop/Embeded project/web pages/combined_html_content.txt'

    # Initialize the classifier
    text_classifier = TextClassifier()

    # Preprocess the data from the combined file
    data_df = text_classifier.preprocess_data(combined_file_path)

    # Train and evaluate the model on the data
    if not data_df.empty:
        text_classifier.train_and_evaluate(data_df)
        
        # Example of making a prediction on a new piece of text
        new_ticket_text = "IT Support Ticket #999 - New User Onboarding. I need assistance with setting up my new laptop and accessing my email account. For additional assistance, please contact the IT Help Desk."
        text_classifier.predict_new_ticket(new_ticket_text)