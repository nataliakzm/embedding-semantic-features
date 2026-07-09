import numpy as np

def get_feature_vector(group1, group2):
    """
    a: list of length n of words on one end of the feature continuum
    b: list of length m of words on the other end of the continuum
    """

    A = group1
    B = group2

    lines = [] # will be a (n x m, 300) matrix
    for A_i in A:
        for B_j in B:
            lines.append(B_j - A_i)

    feature_vector = np.mean(np.array(lines), axis = 0) # shape (300,)
    return feature_vector

def get_projection_score(u, v):
    """
    Get a scalar magnitude of u on v
    """
    projection_score = (np.dot(u, v)/np.dot(v, v))
    return projection_score


# =============================================================================
# Axis Classification Accuracy
# =============================================================================

def axis_classification_accuracy(X_train, y_train, X_test, y_test, axis_vector):
    """
    Use a semantic axis as a binary classifier: project embeddings onto the axis,
    find the optimal threshold on training data, and report test accuracy.

    Args:
        X_train: Training embeddings (n_train, dim)
        y_train: Training labels (n_train,) with values 0 and 1
        X_test: Test embeddings (n_test, dim)
        y_test: Test labels (n_test,)
        axis_vector: Semantic axis vector (dim,)

    Returns:
        dict with train_accuracy, test_accuracy, threshold
    """
    # Project onto axis
    train_scores = np.dot(X_train, axis_vector)
    test_scores = np.dot(X_test, axis_vector)

    # Find optimal threshold on training data (sweep over sorted unique scores)
    # Use midpoints between consecutive sorted scores to avoid bias
    sorted_scores = np.sort(np.unique(train_scores))
    if len(sorted_scores) > 1000:
        # Subsample thresholds for efficiency
        indices = np.linspace(0, len(sorted_scores) - 1, 1000, dtype=int)
        candidates = sorted_scores[indices]
    else:
        candidates = sorted_scores

    best_acc = 0.0
    best_threshold = 0.0
    best_direction = 1  # +1 means class1 > threshold, -1 means class1 < threshold

    for t in candidates:
        for direction in [1, -1]:
            if direction == 1:
                preds = (train_scores >= t).astype(int)
            else:
                preds = (train_scores < t).astype(int)
            acc = np.mean(preds == y_train)
            if acc > best_acc:
                best_acc = acc
                best_threshold = t
                best_direction = direction

    # Apply to test
    if best_direction == 1:
        train_preds = (train_scores >= best_threshold).astype(int)
        test_preds = (test_scores >= best_threshold).astype(int)
    else:
        train_preds = (train_scores < best_threshold).astype(int)
        test_preds = (test_scores < best_threshold).astype(int)

    train_accuracy = float(np.mean(train_preds == y_train))
    test_accuracy = float(np.mean(test_preds == y_test))

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'threshold': float(best_threshold),
        'direction': best_direction,
    }


# =============================================================================
# Separation Metrics (Geometric Separability Index & Mann-Whitney U)
# =============================================================================

def geometric_separability_index(scores_class0, scores_class1):
    """
    Compute the Geometric Separability Index (GSI).

    For each sample, check if its nearest neighbor (by projection score)
    is from the same class. Average across all samples.

    Args:
        scores_class0: 1D array of projection scores for class 0 (common sense)
        scores_class1: 1D array of projection scores for class 1 (non-common sense)

    Returns:
        gsi: float between 0.5 (random) and 1.0 (perfectly separated)

    Interpretation:
        ~1.0 = completely separated (nearest neighbor always same class)
        ~0.5 = weakly separated (nearest neighbor is random)
    """
    scores_class0 = np.asarray(scores_class0)
    scores_class1 = np.asarray(scores_class1)

    # Combine all scores with labels
    all_scores = np.concatenate([scores_class0, scores_class1])
    all_labels = np.concatenate([
        np.zeros(len(scores_class0)),
        np.ones(len(scores_class1))
    ])

    # Sort by score
    sorted_indices = np.argsort(all_scores)
    sorted_labels = all_labels[sorted_indices]

    correct_neighbors = 0
    total = len(all_scores)

    for i in range(total):
        current_label = sorted_labels[i]

        # Find nearest neighbor(s) - check both left and right
        neighbors = []
        if i > 0:
            neighbors.append(sorted_labels[i - 1])
        if i < total - 1:
            neighbors.append(sorted_labels[i + 1])

        if len(neighbors) == 0:
            continue

        # Check if nearest neighbor is same class
        # If two neighbors, use the closest one by score distance
        if len(neighbors) == 1:
            if neighbors[0] == current_label:
                correct_neighbors += 1
        else:
            # Both neighbors exist - check which is closer
            left_dist = all_scores[sorted_indices[i]] - all_scores[sorted_indices[i-1]]
            right_dist = all_scores[sorted_indices[i+1]] - all_scores[sorted_indices[i]]

            if left_dist <= right_dist:
                nearest = sorted_labels[i - 1]
            else:
                nearest = sorted_labels[i + 1]

            if nearest == current_label:
                correct_neighbors += 1

    gsi = correct_neighbors / total
    return gsi


def mann_whitney_u_test(scores_class0, scores_class1):
    """
    Perform Mann-Whitney U test to check if two groups differ statistically.

    Args:
        scores_class0: 1D array of projection scores for class 0
        scores_class1: 1D array of projection scores for class 1

    Returns:
        dict with:
            - statistic: U statistic
            - p_value: p-value (< 0.05 = significant difference)
            - significant: bool indicating if p < 0.05
            - effect_size: rank-biserial correlation (effect size measure)

    Interpretation:
        p_value < 0.05: Groups are statistically different
        effect_size: -1 to 1, where |r| > 0.5 is large effect
    """
    from scipy.stats import mannwhitneyu

    scores_class0 = np.asarray(scores_class0)
    scores_class1 = np.asarray(scores_class1)

    # Perform Mann-Whitney U test (two-sided)
    statistic, p_value = mannwhitneyu(
        scores_class0, scores_class1,
        alternative='two-sided'
    )

    # Compute effect size (rank-biserial correlation)
    n1, n2 = len(scores_class0), len(scores_class1)
    effect_size = 1 - (2 * statistic) / (n1 * n2)

    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size': effect_size
    }


def compute_separation_metrics(embeddings, labels, projection_vector, bias=0.0):
    """
    Compute all separation metrics for embeddings projected onto a vector.

    Args:
        embeddings: 2D array (n_samples, n_features)
        labels: 1D array of class labels (0 or 1)
        projection_vector: 1D array to project embeddings onto
        bias: Optional bias term to add to scores (default: 0.0)
              For NN comparison, use the trained bias to match NN decision boundary.

    Returns:
        dict with all separation metrics
    """
    # Project embeddings onto the vector and add bias
    # This matches the NN computation: score = dot(embedding, weights) + bias
    scores = np.dot(embeddings, projection_vector) + bias

    # Split by class
    scores_class0 = scores[labels == 0]
    scores_class1 = scores[labels == 1]

    # Compute metrics
    gsi = geometric_separability_index(scores_class0, scores_class1)
    mw_results = mann_whitney_u_test(scores_class0, scores_class1)

    # Additional descriptive stats
    mean_diff = np.mean(scores_class1) - np.mean(scores_class0)

    return {
        'geometric_separability_index': gsi,
        'mann_whitney_u': mw_results,
        'mean_score_class0': float(np.mean(scores_class0)),
        'mean_score_class1': float(np.mean(scores_class1)),
        'mean_difference': float(mean_diff),
        'std_class0': float(np.std(scores_class0)),
        'std_class1': float(np.std(scores_class1))
    }