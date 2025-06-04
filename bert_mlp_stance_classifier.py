import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm


MODEL_NAME = "dbmdz/bert-base-italian-uncased"


def compute_embeddings(texts, tokenizer, model, batch_size=16):
    model.eval()
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls)
    return torch.cat(embeddings).numpy()


def main():
    df = pd.read_csv("cleaned_stance_dataset_enriched.csv")
    feature_cols = [
        "EMOJI_COUNT",
        "LINK present",
        "NUM_WORDS",
        "NUM_UPPERCASE",
        "bio_sentiment",
    ]
    df = df[["content", "stance"] + feature_cols].fillna(0)

    texts = df["content"].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    print("Computing BERT embeddings...")
    text_embeddings = compute_embeddings(texts, tokenizer, model)

    numeric_features = df[feature_cols].values
    X = np.hstack([text_embeddings, numeric_features])
    y = df["stance"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=20, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    main()
