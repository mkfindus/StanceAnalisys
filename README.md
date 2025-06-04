# StanceAnalisys
Stance Analisys and Narrative identification

## MLP Stance Classifier
A simple multilayer perceptron (`mlp_stance_classifier.py`) trains on
additional numeric fields from `cleaned_stance_dataset_enriched.csv` to
predict the `stance` label.

```bash
pip install -r requirements.txt
python mlp_stance_classifier.py
```

## BERT + MLP Classifier
`bert_mlp_stance_classifier.py` combines BERT text embeddings with the
same numeric features and trains an MLP on the concatenated vectors.
This allows leveraging contextual information from the post while still
using quantitative metadata like emoji counts or sentiment.

```bash
python bert_mlp_stance_classifier.py
```
