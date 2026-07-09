"""
PCA dimensionality check for embedding space analysis.

Inspired by Gurnee & Tegmark (2024) "Language Models Represent Space and Time"
(ICLR 2024), who run probes on PCA-reduced activations to check how many dimensions
carry the relevant information. Here we PCA the embeddings, retrain the single-layer
classifier at each k, and check whether NN-vs-axis cosine similarity holds in
reduced dimensions. If alignment survives in a low-dimensional subspace, that
strengthens the interpretability claim.
"""

import numpy as np
from sklearn.decomposition import PCA
from src.n_networks.single_layer_nn import SingleLayerNN
from src.analysis.compare_welights import compare_weights_with_axis
from src.analysis.scoring import compute_separation_metrics
from src import logger


def run_pca_analysis(X_train, y_train, X_test, y_test, axis_vectors,
                     k_values=None, training_params=None):
    """
    PCA the embeddings, retrain SL classifier at each k, compare with axes.

    Args:
        X_train: Training embeddings (n_train, dim)
        y_train: Training labels (n_train,)
        X_test: Test embeddings (n_test, dim)
        y_test: Test labels (n_test,)
        axis_vectors: dict {axis_name: axis_vector_full_dim}
        k_values: list of int, PCA components to test (default: [50, 100, 250, 500, 1000, 2000])
        training_params: dict with learning_rate, epochs, batch_size, patience, l2_lambda

    Returns:
        dict with explained_variance_ratios and per_k_results
    """
    if k_values is None:
        k_values = [50, 100, 250, 500, 1000, 2000]

    if training_params is None:
        training_params = {}

    lr = training_params.get('learning_rate', 0.01)
    epochs = training_params.get('epochs', 500)
    batch_size = training_params.get('batch_size', 32)
    patience = training_params.get('patience', 50)
    l2_lambda = training_params.get('l2_lambda', 0)

    max_k = max(k_values)
    input_dim = X_train.shape[1]
    max_k = min(max_k, input_dim, X_train.shape[0])

    logger.info("pca_analysis_starting",
                k_values=k_values,
                max_k=max_k,
                input_dim=input_dim)

    # Fit PCA once with max components needed
    pca = PCA(n_components=max_k)
    X_train_pca_full = pca.fit_transform(X_train)
    X_test_pca_full = pca.transform(X_test)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Project axis vectors into PCA space
    axis_vectors_pca = {}
    for axis_name, axis_vec in axis_vectors.items():
        axis_vectors_pca[axis_name] = pca.transform(axis_vec.reshape(1, -1)).flatten()

    per_k_results = {}

    for k in k_values:
        if k > max_k:
            logger.warning("pca_k_exceeds_max", k=k, max_k=max_k)
            continue

        logger.info("pca_training_at_k", k=k,
                    explained_variance=round(float(cumulative_variance[k - 1]), 4))

        # Slice to k components
        X_train_k = X_train_pca_full[:, :k]
        X_test_k = X_test_pca_full[:, :k]

        # Retrain single-layer NN
        nn_k = SingleLayerNN(input_dim=k)
        nn_k.train(X_train_k, y_train,
                   X_val=X_test_k, y_val=y_test,
                   learning_rate=lr, epochs=epochs,
                   batch_size=batch_size, patience=patience,
                   l2_lambda=l2_lambda, verbose_every=0)

        # Evaluate
        y_pred_train = nn_k.forward(X_train_k)
        y_pred_test = nn_k.forward(X_test_k)
        train_acc = float(nn_k.compute_accuracy(y_pred_train, y_train))
        test_acc = float(nn_k.compute_accuracy(y_pred_test, y_test))
        weights_k = nn_k.get_weights()

        # Separation metrics
        separation = compute_separation_metrics(X_test_k, y_test, weights_k, bias=nn_k.bias)

        # Compare weights vs projected axes
        axis_comparisons = {}
        for axis_name, axis_vec_pca in axis_vectors_pca.items():
            axis_k = axis_vec_pca[:k]
            comparison = compare_weights_with_axis(weights_k, axis_k)
            axis_comparisons[axis_name] = {
                'cosine_similarity': float(comparison['cosine_similarity']),
                'angle_degrees': float(comparison['angle_degrees']),
                'correlation': float(comparison['correlation']),
            }

        per_k_results[k] = {
            'explained_variance_cumulative': float(cumulative_variance[k - 1]),
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'gsi': float(separation['geometric_separability_index']),
            'axis_comparisons': axis_comparisons,
        }

        logger.info("pca_result_at_k",
                    k=k,
                    test_accuracy=round(test_acc, 4),
                    gsi=round(float(separation['geometric_separability_index']), 4),
                    explained_variance=round(float(cumulative_variance[k - 1]), 4))

    results = {
        'k_values': k_values,
        'per_k': per_k_results,
    }

    return results
