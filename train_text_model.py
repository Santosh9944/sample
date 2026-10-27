import os
import re
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_and_save_model(csv_path="test_news.csv", model_path="models/text_model.joblib"):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path).dropna(subset=["text", "label"])
    df["label"] = df["label"].str.strip().str.upper()
    df = df[df["label"].isin(["FAKE", "REAL"])]
    print(f"Dataset samples: {len(df)}")
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=2.0,
            class_weight="balanced"
        ))
    ])
    
    print("Training TF-IDF + LogisticRegression model...")
    pipeline.fit(df["text"], df["label"])
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved successfully to {model_path}")
    return pipeline

if __name__ == "__main__":
    train_and_save_model()
