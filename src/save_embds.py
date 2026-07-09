import pandas as pd

def save_embeddings_to_excel(embeddings, sentences, labels, output_path, prefix=""):
    """
    Save embeddings to Excel format

    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        sentences: list of sentence strings
        labels: numpy array of labels
        output_path: Path to output directory
        prefix: prefix for filename (e.g., 'train' or 'test')
    """
    hidden_size = embeddings.shape[1]
    embedding_columns = [f"hidden_state_{i}" for i in range(hidden_size)]

    df = pd.DataFrame({
        'sentence': sentences,
        'label': labels
    })

    # Add embedding columns
    embedding_df = pd.DataFrame(embeddings, columns=embedding_columns)
    df = pd.concat([df.reset_index(drop=True), embedding_df.reset_index(drop=True)], axis=1)

    # Save to Excel
    output_file = output_path / f"{prefix}_embeddings.xlsx"
    df.to_excel(output_file, index=False)

    return output_file
