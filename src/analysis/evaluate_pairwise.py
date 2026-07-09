import numpy as np
from src.analysis.scoring import get_projection_score

def evaluate_pairwise(pairs, embeddings_dict, nn_weights, axis_vector, bias=0.0):
    """
    Evaluate both NN and semantic axis using pairwise comparison.

    Args:
        pairs: List of (non_common_sense, common_sense) sentence tuples
        embeddings_dict: Dict mapping sentence -> embedding
        nn_weights: Trained NN weight vector
        axis_vector: Semantic axis vector
        bias: NN bias term

    Returns:
        Dictionary with pairwise evaluation results
    """
    nn_correct = 0
    axis_correct = 0
    total_pairs = 0
    missing_pairs = 0

    for non_cs_sentence, cs_sentence in pairs:
        # Get embeddings (normalize keys to match)
        non_cs_key = non_cs_sentence.lower().strip()
        cs_key = cs_sentence.lower().strip()

        non_cs_emb = embeddings_dict.get(non_cs_key)
        cs_emb = embeddings_dict.get(cs_key)

        if non_cs_emb is None or cs_emb is None:
            missing_pairs += 1
            continue

        total_pairs += 1

        # NN scores (higher = more likely non-common-sense)
        nn_non_cs_score = np.dot(non_cs_emb, nn_weights) + bias
        nn_cs_score = np.dot(cs_emb, nn_weights) + bias

        if nn_non_cs_score > nn_cs_score:
            nn_correct += 1

        # Semantic axis scores (projection onto axis)
        axis_non_cs_score = get_projection_score(non_cs_emb, axis_vector)
        axis_cs_score = get_projection_score(cs_emb, axis_vector)

        if axis_non_cs_score > axis_cs_score:
            axis_correct += 1

    nn_accuracy = nn_correct / total_pairs if total_pairs > 0 else 0
    axis_accuracy = axis_correct / total_pairs if total_pairs > 0 else 0

    return {
        'total_pairs': total_pairs,
        'missing_pairs': missing_pairs,
        'nn_correct': nn_correct,
        'nn_accuracy': nn_accuracy,
        'axis_correct': axis_correct,
        'axis_accuracy': axis_accuracy,
        'nn_advantage': nn_accuracy - axis_accuracy
    }

