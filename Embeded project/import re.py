import os
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

class TextClassifier:
    def __init__(self, model_name="distilbert-base-uncased"):
        print("Loading Hugging Face tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = None
        self.label_mapping = {}
        self.reverse_label_mapping = {}

    def preprocess_data(self, file_path):
        print(f"Reading and preprocessing data from '{file_path}'...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                full_content = f.read()
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            return pd.DataFrame()

        rows = []
        # Correct regex for "Ticket #001 - System Crash"
        ticket_pattern = re.findall(r"Ticket\s*#\s*(\d+)\s*-\s*([^\n\r]+)", full_content)

        for ticket_num, issue_text in ticket_pattern:
            issue_text = issue_text.lower().strip()
            if "database" in issue_text:
                y_label = "database crash"
            elif "software" in issue_text or "application" in issue_text:
                y_label = "software crash"
            elif "server" in issue_text or "power supply" in issue_text:
                y_label = "server crash"
            elif ("system" in issue_text
                  or "cpu" in issue_text
                  or "memory" in issue_text
                  or "blue screen" in issue_text
                  or "thermal" in issue_text):
                y_label = "system crash"
            else:
                y_label = "other issue"
            rows.append({"x_text": f"Ticket #{ticket_num}", "y_label": y_label})

        df = pd.DataFrame(rows)
        print("\n✅ Preprocessed DataFrame:")
        print(df.head(20))
        return df

    def create_embeddings(self, texts):
        print("Generating text embeddings...")
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def train_and_evaluate(self, df):
        if df.empty:
            print("No data to train or evaluate.")
            return

        unique_labels = df['y_label'].unique()
        self.label_mapping = {label: i for i, label in enumerate(unique_labels)}
        self.reverse_label_mapping = {i: label for label, i in self.label_mapping.items()}
        y_labels = df['y_label'].map(self.label_mapping)

        embeddings = self.create_embeddings(df['x_text'].tolist())
        X_train = embeddings.numpy()
        y_train = y_labels.tolist()

        print("\nTraining XGBoost model...")
        self.classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.classifier.fit(X_train, y_train)

        print("\n--- Model Evaluation ---")
        y_pred = self.classifier.predict(X_train)
        print(f"Accuracy: {accuracy_score(y_train, y_pred):.2f}")
        print("\nClassification Report:")
        print(classification_report(y_train, y_pred, target_names=self.label_mapping.keys()))

    def predict_new_ticket(self, new_text):
        if self.classifier is None:
            print("Error: The model has not been trained yet.")
            return

        print("\n--- Predicting New Ticket ---")
        new_embedding = self.create_embeddings([new_text]).numpy()
        prediction = self.classifier.predict(new_embedding)
        predicted_label = self.reverse_label_mapping[prediction[0]]

        print(f"New Ticket Text: '{new_text}'")
        print(f"Predicted Category: {predicted_label}")

if __name__ == "__main__":
    # Path to your single file with 36 tickets
    file_path = 'C:/Users/DELL/Desktop/Embeded project/web pages/combined_html_content.txt'

    text_classifier = TextClassifier()
    data_df = text_classifier.preprocess_data(file_path)
    if not data_df.empty:
        text_classifier.train_and_evaluate(data_df)
        # Example prediction
        new_ticket_text = "Ticket #999 - Server overheating and shutting down"
        text_classifier.predict_new_ticket(new_ticket_text)
    else:
        print("No valid data found to process.")