import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Sample ground truth: list of similar JSON pairs
ground_truth = [(0, 1), (0, 2), (1, 2), ...]

# Function to calculate precision, recall, and F1-score
def evaluate_precision_recall_f1(predicted_similarities, ground_truth):
    y_true = np.zeros(len(predicted_similarities))
    y_pred = np.zeros(len(predicted_similarities))
    
    for pair in ground_truth:
        y_true[pair[0], pair[1]] = 1
        y_true[pair[1], pair[0]] = 1
        
    for pair in predicted_similarities:
        y_pred[pair[0], pair[1]] = 1
        y_pred[pair[1], pair[0]] = 1

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return precision, recall, f1

# Function to calculate Mean Reciprocal Rank (MRR)
def mean_reciprocal_rank(y_true, y_pred):
    rs = (np.asarray(r).nonzero()[0] for r in y_pred)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

# Function to calculate Normalized Discounted Cumulative Gain (NDCG)
def ndcg_at_k(y_true, y_pred, k=10):
    dcg = 0
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(k, np.count_nonzero(y_true)))])
    
    for i, rel in enumerate(y_pred[:k]):
        if rel:
            dcg += 1.0 / np.log2(i + 2)
    
    return dcg / idcg

# Example Usage:
predicted_similarities_kmeans = [(0, 1), (1, 3), ...]
predicted_similarities_autoencoder = [(0, 2), (1, 2), ...]
predicted_similarities_word2vec = [(0, 1), (1, 2), ...]

precision_kmeans, recall_kmeans, f1_kmeans = evaluate_precision_recall_f1(predicted_similarities_kmeans, ground_truth)
precision_autoencoder, recall_autoencoder, f1_autoencoder = evaluate_precision_recall_f1(predicted_similarities_autoencoder, ground_truth)
precision_word2vec, recall_word2vec, f1_word2vec = evaluate_precision_recall_f1(predicted_similarities_word2vec, ground_truth)

mrr_kmeans = mean_reciprocal_rank(ground_truth, predicted_similarities_kmeans)
mrr_autoencoder = mean_reciprocal_rank(ground_truth, predicted_similarities_autoencoder)
mrr_word2vec = mean_reciprocal_rank(ground_truth, predicted_similarities_word2vec)

ndcg_kmeans = ndcg_at_k(ground_truth, predicted_similarities_kmeans)
ndcg_autoencoder = ndcg_at_k(ground_truth, predicted_similarities_autoencoder)
ndcg_word2vec = ndcg_at_k(ground_truth, predicted_similarities_word2vec)

print("K-Means: Precision:", precision_kmeans, "Recall:", recall_kmeans, "F1-Score:", f1_kmeans, "MRR:", mrr_kmeans, "NDCG:", ndcg_kmeans)
print("Autoencoder: Precision:", precision_autoencoder, "Recall:", recall_autoencoder, "F1-Score:", f1_autoencoder, "MRR:", mrr_autoencoder, "NDCG:", ndcg_autoencoder)
print("Word2Vec: Precision:", precision_word2vec, "Recall:", recall_word2vec, "F1-Score:", f1_word2vec, "MRR:", mrr_word2vec, "NDCG:", ndcg_word2vec)

