"""
Ridge regression probe for comparing learned directions.

Inspired by Gurnee & Tegmark (2024) "Language Models Represent Space and Time"
(ICLR 2024), who use linear ridge regression probes on frozen LLM representations
to extract interpretable directions. Here we train a ridge classifier on the same
embeddings as the NN, extract its weight vector, and compare it against both the
NN weights and hand-crafted semantic axes -- a three-way directional comparison.
"""

import numpy as np
from sklearn.linear_model import RidgeClassifier
from src.analysis.compare_welights import compare_weights_with_axis, compare_rank_order
from src.analysis.scoring import compute_separation_metrics
from src import logger


def run_ridge_probe(X_train, y_train, X_test, y_test,
                    nn_weights, nn_bias, axis_vectors, alpha=1.0):
    """
    Train ridge classifier, extract weights, compare against NN and axes.

    Args:
        X_train: Training embeddings (n_train, dim)
        y_train: Training labels (n_train,)
        X_test: Test embeddings (n_test, dim)
        y_test: Test labels (n_test,)
        nn_weights: Learned NN weight vector (dim,)
        nn_bias: NN bias scalar
        axis_vectors: dict {axis_name: axis_vector}
        alpha: Ridge regularization strength

    Returns:
        dict with ridge results, comparisons, and weight vector
    """
    logger.info("training_ridge_probe", alpha=alpha)

    ridge = RidgeClassifier(alpha=alpha)
    ridge.fit(X_train, y_train)

    ridge_weights = ridge.coef_.flatten()
    ridge_bias = float(ridge.intercept_[0]) if hasattr(ridge.intercept_, '__len__') else float(ridge.intercept_)
    ridge_accuracy = float(ridge.score(X_test, y_test))

    logger.info("ridge_probe_trained",
                test_accuracy=round(ridge_accuracy, 4),
                weight_mean=round(float(np.mean(ridge_weights)), 4),
                weight_std=round(float(np.std(ridge_weights)), 4))

    # Ridge vs NN weight comparison (skip if nn_weights is zeros, e.g. ML 4096-dim)
    has_nn_weights = np.any(nn_weights != 0)
    ridge_scores = X_test @ ridge_weights + ridge_bias

    if has_nn_weights:
        ridge_vs_nn = compare_weights_with_axis(ridge_weights, nn_weights)
        logger.info("ridge_vs_nn",
                    cosine_similarity=round(float(ridge_vs_nn['cosine_similarity']), 4),
                    angle_degrees=round(float(ridge_vs_nn['angle_degrees']), 2),
                    correlation=round(float(ridge_vs_nn['correlation']), 4))

        nn_scores = X_test @ nn_weights + nn_bias
        ridge_vs_nn_rank = compare_rank_order(ridge_scores, nn_scores, labels=y_test)
        logger.info("ridge_vs_nn_rank_order",
                    spearman_correlation=round(ridge_vs_nn_rank['spearman_correlation'], 4),
                    mean_displacement=round(ridge_vs_nn_rank['mean_displacement'], 2))
    else:
        ridge_vs_nn = {}
        ridge_vs_nn_rank = {}

    # Ridge separation metrics
    ridge_separation = compute_separation_metrics(X_test, y_test, ridge_weights, bias=ridge_bias)
    logger.info("ridge_separation_metrics",
                gsi=round(ridge_separation['geometric_separability_index'], 4),
                effect_size=round(ridge_separation['mann_whitney_u']['effect_size'], 4))

    # Ridge vs each semantic axis
    per_axis = {}
    for axis_name, axis_vector in axis_vectors.items():
        comparison = compare_weights_with_axis(ridge_weights, axis_vector)
        per_axis[axis_name] = {
            'cosine_similarity': float(comparison['cosine_similarity']),
            'angle_degrees': float(comparison['angle_degrees']),
            'correlation': float(comparison['correlation']),
        }
        logger.info("ridge_vs_axis",
                    axis_name=axis_name,
                    cosine_similarity=round(float(comparison['cosine_similarity']), 4),
                    angle_degrees=round(float(comparison['angle_degrees']), 2))

    results = {
        'alpha': alpha,
        'test_accuracy': ridge_accuracy,
        'separation_metrics': {
            'geometric_separability_index': float(ridge_separation['geometric_separability_index']),
            'mann_whitney_p_value': float(ridge_separation['mann_whitney_u']['p_value']),
            'effect_size': float(ridge_separation['mann_whitney_u']['effect_size']),
        },
        'per_axis': per_axis,
    }

    if has_nn_weights:
        results['ridge_vs_nn'] = {
            'cosine_similarity': float(ridge_vs_nn['cosine_similarity']),
            'angle_degrees': float(ridge_vs_nn['angle_degrees']),
            'correlation': float(ridge_vs_nn['correlation']),
        }
        results['ridge_vs_nn_rank_order'] = {
            'spearman_correlation': float(ridge_vs_nn_rank['spearman_correlation']),
            'spearman_pvalue': float(ridge_vs_nn_rank['spearman_pvalue']),
            'mean_displacement': float(ridge_vs_nn_rank['mean_displacement']),
        }

    return results, ridge_weights
