import numpy as np

def get_feature_vector(group1, group2):
    """
    Computes the average difference vector (feature vector) between two groups of embeddings.

    Args:
        group1: List or array of embeddings representing one end of a feature continuum.
        group2: List or array of embeddings representing the other end of the continuum.

    Returns:
        A numpy array representing the mean difference vector between all pairs (group2_i - group1_j).
    """
    A = group1
    B = group2

    lines = []
    for A_i in A:
        for B_j in B:
            lines.append(B_j - A_i)

    feature_vector = np.mean(np.array(lines), axis = 0)
    return feature_vector

def get_projection_score(u, v):
    """
    Computes the scalar projection of vector u onto vector v.

    Args:
        u: The vector to be projected (numpy array).
        v: The vector to project onto (numpy array).

    Returns:
        Scalar value representing the magnitude of u in the direction of v.
    """
    projection_score = (np.dot(u, v)/np.dot(v, v))
    return projection_score

def get_scores(s_embeddings, group1, group2):
    """
    Projects each sentence embedding onto the feature vector defined by two groups.

    Args:
        s_embeddings: List or array of sentence embeddings to score.
        group1: List or array of embeddings for one end of the feature continuum.
        group2: List or array of embeddings for the other end of the continuum.

    Returns:
        List of scalar projection scores for each sentence embedding.
    """
    # get feature subspace
    feature_vector = get_feature_vector(group1, group2)

    # get projection scores
    projection_scores = [get_projection_score(sentence, feature_vector) for sentence in s_embeddings]

    return projection_scores