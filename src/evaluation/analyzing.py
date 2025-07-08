from src.evaluation.comparison import *
from src.evaluation.scoring import get_scores
from src.utils import logger, send_slack_notification

__all__ = ["analysing"]

def analysing(
    sentences_v1_embeddings, 
    safe_group1_embeddings, 
    danger_group2_embeddings, 
    sentence_category, 
    METHOD
    ):
    
    logger.info("\n=== Starting Analysis ===")
    
    if sentence_category == 'danger':
        sentence_list = [sentence.lower() for sentence in danger_category.split(", ")]
        normalized_pairs = [(dangerous.lower().strip(), safer.lower().strip()) for dangerous, safer in danger_pairs]
        
    # ---------------------------------------------------------------------------------------------------------------
    
    elif sentence_category == 'size':
        sentence_list = [sentence.lower() for sentence in size_category.split(", ")]
        normalized_pairs = [(dangerous.lower().strip(), safer.lower().strip()) for dangerous, safer in size_pairs]
    
    else:
        raise ValueError(f"Unknown script: {sentence_category}")
    
    semantic_projection_scores = get_scores(sentences_v1_embeddings, safe_group1_embeddings, danger_group2_embeddings)
    
    # Create a dictionary mapping sentences to their scores
    scores_dict = {sentence: score for sentence, score in zip(sentence_list, semantic_projection_scores)}

    # Evaluate if the model rates the more dangerous sentence higher
    correct_count = 0
    total_pairs = len(normalized_pairs)

    for dangerous_sentence, safer_sentence in normalized_pairs:
        if scores_dict[dangerous_sentence] > scores_dict[safer_sentence]:
            correct_count += 1

    # Calculate the fraction of correct responses
    correct_fraction = correct_count / total_pairs
    print("------------------------")
    print(f"RESULTS: Correctly rated pairs: {correct_count}/{total_pairs} = {correct_fraction:.2f}")
    logger.warning(f"Correctly rated pairs: {correct_count}/{total_pairs}", RESULT=f"{correct_fraction:.2f}", METHOD=f"{METHOD}")
    send_slack_notification(
        f"Correctly rated pairs: {correct_count}/{total_pairs} = {correct_fraction:.2f} for {sentence_category} using {METHOD}"
    )
    print("------------------------")
    logger.info("✓ Analysis completed successfully")
    return correct_fraction
    
