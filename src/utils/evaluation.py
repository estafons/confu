import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import csv
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegressionCV





def recall_at_k(modality1_embeddings, modality2_embeddings, ks=[1,5,10], modalities=None):
    """
    Compute recall@K for cross-modal retrieval both directions.

    Args:
        modality1_embeddings (torch.Tensor): shape [N, D]
        modality2_embeddings (torch.Tensor): shape [N, D]
        ks (list[int]): list of k values to compute recall@k

    Returns:
        dict: {"m1->m2_recall@k": val, "m2->m1_recall@k": val}
    """

    total = len(modality1_embeddings)
    print(f"Computing recall@k for {total} samples.")
    modality1_embeddings = torch.tensor(modality1_embeddings, dtype=torch.float32)
    modality2_embeddings = torch.tensor(modality2_embeddings, dtype=torch.float32)

    
    # similarity matrix: [N, N]
    sims = modality1_embeddings @ modality2_embeddings.T 
    N = sims.size(0)

    results = {}

    # m1 -> m2 retrieval
    ranks = sims.argsort(dim=1, descending=True)  # [N, N]
    gt = torch.arange(N).unsqueeze(1)  # ground truth indices
    for k in ks:
        correct = (ranks[:, :k] == gt).any(dim=1).float().mean().item()
        results[f"M{modalities[0]}->_M{modalities[1]}_recall@{k}"] = correct

    # m2 -> m1 retrieval
    ranks = sims.T.argsort(dim=1, descending=True)  # [N, N]
    for k in ks:
        correct = (ranks[:, :k] == gt).any(dim=1).float().mean().item()
        results[f"M{modalities[1]}->_M{modalities[0]}_recall@{k}"] = correct

    return results


def train_evaluate_logistic(train_samples, train_labels, test_samples, test_labels, txt=None):
    """
    Train logistic regression on frozen embeddings and evaluate test accuracy.
    """
    # Flatten embeddings if needed
    train_embeddings = train_samples
    test_embeddings = test_samples

    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_embeddings, train_labels.cpu().numpy())

    predictions = clf.predict(test_embeddings)
    accuracy = accuracy_score(test_labels.cpu().numpy(), predictions)

    print(f"Logistic Regression Test Accuracy for {txt}: {accuracy:.4f}")
    return accuracy

def comm_classification_scoring(train_samples, train_labels, test_samples, test_labels, scoring="balanced_accuracy", txt=None):
    linear_model = LogisticRegressionCV(Cs=5, n_jobs=10, scoring=scoring)
    linear_model.fit(train_samples.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
    score = linear_model.score(test_samples.cpu().detach().numpy(), test_labels.cpu().detach().numpy())
    print(f"Logistic Regression Test Accuracy for {txt}: {score:.4f}")
    return score


def train_evaluate_mlp(train_samples, train_labels, test_samples, test_labels, txt=None, hidden_dim=256, epochs=20, lr=1e-3, batch_size=64):
    """
    Train a two-layer MLP on frozen embeddings and evaluate test accuracy.
    """

    # ensure tensors are floats
    train_samples = torch.tensor(train_samples, dtype=torch.float32)
    test_samples = torch.tensor(test_samples, dtype=torch.float32)
    train_labels = train_labels.long()
    test_labels = test_labels.long()

    input_dim = train_samples.shape[1]
    num_classes = len(torch.unique(train_labels))

    # simple 2-layer MLP
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training loop
    for epoch in range(epochs):
        perm = torch.randperm(train_samples.size(0))
        for i in range(0, train_samples.size(0), batch_size):
            idx = perm[i:i+batch_size]
            x_batch = train_samples[idx]
            y_batch = train_labels[idx]

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    # evaluation
    with torch.no_grad():
        logits = model(test_samples)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    accuracy = accuracy_score(test_labels.cpu().numpy(), preds)
    print(f"MLP Test Accuracy for {txt}: {accuracy:.4f}")
    return accuracy


def write_results_to_file(results, cfg, prefix=''):
    """
    Write results to a text file.
    """
    # os.makedirs(os.path.dirname(cfg.results_path), exist_ok=True)
    os.makedirs(cfg.results_path, exist_ok=True)

    # Define CSV file name based on cfg
    # if cfg.lambda_ is not None:
    #     csv_filename = f"{cfg.results_path}/{prefix}{cfg.scenario}-{cfg.dataset.train_dataset}.lambda_{cfg.lambda_}.mask_ratio_{cfg.mask_ratio}.csv"
    # else:
    csv_filename = f"{cfg.results_path}/{prefix}{cfg.scenario}-{cfg.dataset.train_dataset}.csv"

    print(csv_filename)
    # Write or append to CSV
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"Appended results to {csv_filename}")


def effective_rank(X):
    """
    Compute the effective rank of an embedding matrix X.
    X: shape (n_samples, embedding_dim)
    """
    # Covariance matrix
    C = np.cov(X, rowvar=False)
    
    # Singular values
    s = np.linalg.svd(C, compute_uv=False)
    
    # Normalize
    p = s / s.sum()
    
    # Shannon entropy
    H = -np.sum(p * np.log(p + 1e-12))
    
    # Effective rank
    return np.exp(H)
