import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score


def main():
    df = pd.read_csv('cleaned_stance_dataset_enriched.csv')
    features = ['EMOJI_COUNT', 'LINK present', 'NUM_WORDS', 'NUM_UPPERCASE', 'bio_sentiment']
    X = df[features].fillna(0)
    y = df['stance']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))


if __name__ == '__main__':
    main()
