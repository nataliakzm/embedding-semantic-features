import os
import numpy as np
import pandas as pd

from src.utils import logger

__all__ = ["loading"]

def read_excel_sheets(file_path):
    """Reads all sheets from an Excel file and returns a dictionary with sheet names as keys."""
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return {}
    
    xl = pd.ExcelFile(file_path)
    sheets = {}
    for sheet_name in xl.sheet_names:
        sheets[sheet_name] = xl.parse(sheet_name)
    return sheets


def load_vectors_from_sheet(file_path, sheet_name):
    """Loads vectors from a specified sheet in an Excel file."""
    sheets = read_excel_sheets(file_path)
    vectors = {}
    if sheet_name in sheets:
        df = sheets[sheet_name]
        df = df.drop(columns=['attention_mask', 'input_ids', 'token_type_ids'], errors='ignore')
        for _, row in df.iterrows():
            word = row.iloc[0]
            embedding = row.iloc[1:].values.astype(float)
            vectors[word] = embedding
    else:
        print(f"Sheet {sheet_name} not found in the file.")
    return vectors


def load_vectors_from_sheet_with_duplicates(file_path, sheet_name):
    """Loads vectors from a specified sheet in an Excel file, preserving duplicates."""
    sheets = read_excel_sheets(file_path)
    vectors_list = []
    if sheet_name in sheets:
        df = sheets[sheet_name]
        df = df.drop(columns=['attention_mask', 'input_ids', 'token_type_ids'], errors='ignore')
        for _, row in df.iterrows():
            word = row.iloc[0]
            embedding = row.iloc[1:].values.astype(float)
            vectors_list.append((word, embedding))
    else:
        print(f"Sheet {sheet_name} not found in the file.")
    return vectors_list


def get_embedding_list(vectors):
    """Extracts and returns the list of embedding arrays from the vectors dictionary."""
    return list(vectors.values())


def get_embedding_list_from_tuples(vectors_list):
    """Extracts and returns the list of embedding arrays from a list of (word, embedding) tuples."""
    return [embedding for _, embedding in vectors_list]


def loading(file_path_sentences, file_path_safe, file_path_danger):
    """
    Loads vectors from specified Excel files and sheets and converts them to embedding lists.
    
    Args:
        file_path_sentences: Path to the file containing sentence embeddings
        file_path_safe: Path to the file containing safe group embeddings
        file_path_danger: Path to the file containing danger group embeddings
        
    Returns:
        Tuple containing three lists of embeddings: (sentences, safe_group, danger_group)
    """
    
    logger.info("\n=== Starting Loading ===")

    # Check if we're dealing with size dataset by looking at the file path
    is_size = 'size' in file_path_sentences or 'size' in file_path_sentences

    # Load vectors from the specified sheets
    np.set_printoptions(precision=6, suppress=True)
    
    if is_size:
        logger.info("Detected size dataset, using duplicate-preserving loader")
        extended_sentences_v1 = load_vectors_from_sheet_with_duplicates(file_path_sentences, 'Sheet1')
        safe_group1 = load_vectors_from_sheet_with_duplicates(file_path_safe, 'Sheet1')
        danger_group2 = load_vectors_from_sheet_with_duplicates(file_path_danger, 'Sheet1')
        
        # Convert the vectors to lists of embeddings
        sentences_v1_embeddings = get_embedding_list_from_tuples(extended_sentences_v1)
        safe_group1_embeddings = get_embedding_list_from_tuples(safe_group1)
        danger_group2_embeddings = get_embedding_list_from_tuples(danger_group2)
    else:
        extended_sentences_v1 = load_vectors_from_sheet(file_path_sentences, 'Sheet1')
        safe_group1 = load_vectors_from_sheet(file_path_safe, 'Sheet1')
        danger_group2 = load_vectors_from_sheet(file_path_danger, 'Sheet1')
        
        # Convert the vectors to lists of embeddings
        sentences_v1_embeddings = get_embedding_list(extended_sentences_v1)
        safe_group1_embeddings = get_embedding_list(safe_group1)
        danger_group2_embeddings = get_embedding_list(danger_group2)
    
    return (sentences_v1_embeddings, safe_group1_embeddings, danger_group2_embeddings)