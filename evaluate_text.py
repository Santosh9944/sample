import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load the prepared dataset
df = pd.read_csv("test_news.csv")

df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)
df["label"] = df["label"].str.strip().str.upper()

# Keep valid labels only
df = df[df["label"].isin(["FAKE", "REAL"])]

X = df["text"]
y = df["label"]

print("Total records:", len(df))
print("\nDataset distribution:")
print(y.value_counts())

# 2. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Create and train the baseline model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

print("\nTraining model...")
model.fit(X_train, y_train)

# 4. Predict test data
y_pred = model.predict(X_test)

# 5. Calculate evaluation metrics
print("\n========== MODEL EVALUATION ==========")

print(f"Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred, pos_label='FAKE') * 100:.2f}%")
print(f"Recall   : {recall_score(y_test, y_pred, pos_label='FAKE') * 100:.2f}%")
print(f"F1-Score : {f1_score(y_test, y_pred, pos_label='FAKE') * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 6. Display confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["FAKE", "REAL"],
    yticklabels=["FAKE", "REAL"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 7. Save evaluation results
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

results.to_csv("evaluation_predictions.csv", index=False)

print("\nFiles saved:")
print("1. confusion_matrix.png")
print("2. evaluation_predictions.csv")
