# Extractive Top-K Methods

This dir contains notebooks and scripts for training top-k sentence ranking models from the paper

- `textrank_rouge_comp.py` for the unsupservised TextRank model
- `lstm_model` for the Word2Vec / S-BERT --> Bi-LSTM supervised sentence ranker. 
The optimal sentence ranking 'labels' are inferred directly from the reference summary
