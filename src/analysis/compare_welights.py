import numpy as np
from scipy.stats import spearmanr, rankdata

def compare_weights_with_axis(nn_weights, axis_vector):
    """
    Compare neural network weights with semantic axis

    Args:
        nn_weights: Weights from trained neural network
        axis_vector: Semantic axis from get_feature_vector

    Returns:
        Dictionary with comparison metrics
    """
    # Normalize both vectors
    nn_norm = nn_weights / np.linalg.norm(nn_weights)
    axis_norm = axis_vector / np.linalg.norm(axis_vector)

    # Compute cosine similarity
    cosine_sim = np.dot(nn_norm, axis_norm)

    # Compute angle in degrees
    angle = np.arccos(np.clip(cosine_sim, -1.0, 1.0)) * 180 / np.pi

    # Compute correlation
    correlation = np.corrcoef(nn_weights, axis_vector)[0, 1]

    return {
        'cosine_similarity': cosine_sim,
        'angle_degrees': angle,
        'correlation': correlation
    }


def compare_rank_order(nn_scores, axis_scores, labels=None):
    """
    Compare NN and semantic axis by ranking all test sentences and measuring
    how much their orderings agree. Works regardless of dimensionality since
    inputs are scalar scores.

    Args:
        nn_scores: 1D array of NN scores per sentence (dot product + bias)
        axis_scores: 1D array of axis projection scores per sentence
        labels: Optional 1D array of class labels (0 or 1) for per-class breakdown

    Returns:
        Dictionary with:
            - rank_displacements: per-sentence |nn_rank - axis_rank|
            - mean_displacement: average rank shift
            - median_displacement: median rank shift
            - max_displacement: worst-case rank shift
            - spearman_correlation: Spearman rho between the two rankings
            - spearman_pvalue: p-value for Spearman test
            - matrix_correlation: Pearson correlation between NN and axis
                                  pairwise rank-distance matrices
            - mean_displacement_class0/class1: per-class breakdown (if labels given)
    """
    nn_scores = np.asarray(nn_scores).ravel()
    axis_scores = np.asarray(axis_scores).ravel()

    # Rank sentences (1-based, average ties)
    nn_ranks = rankdata(nn_scores, method='average')
    axis_ranks = rankdata(axis_scores, method='average')

    # Per-sentence displacement
    displacements = np.abs(nn_ranks - axis_ranks)

    # Spearman rank correlation
    sp_corr, sp_pval = spearmanr(nn_scores, axis_scores)

    # Pairwise rank-distance matrices
    nn_dist = np.abs(nn_ranks[:, None] - nn_ranks[None, :])
    axis_dist = np.abs(axis_ranks[:, None] - axis_ranks[None, :])

    # Correlate upper triangles (exclude diagonal)
    triu_idx = np.triu_indices(len(nn_scores), k=1)
    nn_flat = nn_dist[triu_idx]
    axis_flat = axis_dist[triu_idx]
    matrix_corr = np.corrcoef(nn_flat, axis_flat)[0, 1]

    results = {
        'rank_displacements': displacements,
        'nn_ranks': nn_ranks,
        'axis_ranks': axis_ranks,
        'mean_displacement': float(np.mean(displacements)),
        'median_displacement': float(np.median(displacements).item()),
        'max_displacement': int(np.max(displacements).item()),
        'spearman_correlation': float(sp_corr),
        'spearman_pvalue': float(sp_pval),
        'matrix_correlation': float(matrix_corr),
    }

    if labels is not None:
        labels = np.asarray(labels).ravel()
        results['mean_displacement_class0'] = float(np.mean(displacements[labels == 0]))
        results['mean_displacement_class1'] = float(np.mean(displacements[labels == 1]))

    return results