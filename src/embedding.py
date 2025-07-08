import numpy as np
import pandas as pd
import os

from src.utils import logger

def embed_sentences(dat_sentence, METHOD, output_prefix="model"):
    
    #--------------------START----------------------- 
    # If dat_sentence is a list of dicts, convert to DataFrame
    if isinstance(dat_sentence, list) and isinstance(dat_sentence[0], dict):
        dat_sentence = pd.DataFrame(dat_sentence)
            
    # Flatten hidden_state if it's shape (1, N)
    if isinstance(dat_sentence, pd.DataFrame):
            dat_sentence['hidden_state'] = dat_sentence['hidden_state'].apply(
                lambda x: x.flatten() if hasattr(x, 'shape') and len(x.shape) == 2 and x.shape[0] == 1 else x
            )
    logger.info("✓ Sentence embeddings extracted", shape=dat_sentence['hidden_state'].shape)
    df_sentence = dat_sentence if isinstance(dat_sentence, pd.DataFrame) else dat_sentence.to_pandas()
    #---------------------END-----------------------

    # find out the size of each embedding vector
    hidden_size = df_sentence['hidden_state'].iloc[0].shape[0]
    logger.info("✓ Embedding dimension:", hidden_size=hidden_size)

    # Create column names for the embeddings
    embedding_columns = [f"hidden_state_{i}" for i in range(hidden_size)]

    # Expand the hidden_state arrays into separate columns
    embedding_values = np.vstack(df_sentence['hidden_state'].values)
    embedding_df = pd.DataFrame(embedding_values, columns=embedding_columns)

    # Drop the original hidden_state column
    df_sentence = df_sentence.drop(columns=['hidden_state'])
    df_sentence = pd.concat([df_sentence.reset_index(drop=True), embedding_df.reset_index(drop=True)], axis=1)
    output_file = f'./src/data/{output_prefix}_model_{METHOD}.xlsx'
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_sentence.to_excel(output_file, index=False)
    logger.info("✓ Sentence embeddings saved to:", output_file=output_file)
    
    return output_file

def embed_group_1(dat_group1, output_prefix=""):
    
    #--------------------START----------------------- 
    # If dat_group1 is a list of dicts, convert to DataFrame
    if isinstance(dat_group1, list) and isinstance(dat_group1[0], dict):
            dat_group1 = pd.DataFrame(dat_group1)
    # Flatten hidden_state if it's shape (1, N)
    if isinstance(dat_group1, pd.DataFrame):
            dat_group1['hidden_state'] = dat_group1['hidden_state'].apply(
                lambda x: x.flatten() if hasattr(x, 'shape') and len(x.shape) == 2 and x.shape[0] == 1 else x
            )
    logger.info("\n=== Processing Group 1 (Safe Words) ===")
    logger.info("✓ Group 1 embeddings extracted - Shape:", shape=dat_group1['hidden_state'].shape)
    df_group1 = dat_group1 if isinstance(dat_group1, pd.DataFrame) else dat_group1.to_pandas()
    #---------------------END-----------------------

   # find out the size of each embedding vector
    hidden_size_group1 = df_group1['hidden_state'].iloc[0].shape[0]
    logger.info("Hidden size:", hidden_size=hidden_size_group1)

    #Expand the Embeddings into Separate Columns
    embedding_columns = [f"hidden_state_{i}" for i in range(hidden_size_group1)]

    # Expand the hidden_state arrays into separate columns
    embedding_values = np.vstack(df_group1['hidden_state'].values)
    embedding_df_group1 = pd.DataFrame(embedding_values, columns=embedding_columns)

    # Drop the original hidden_state column
    df_group1 = df_group1.drop(columns=['hidden_state'])

    # Concatenate the embeddings DataFrame with the original DataFrame
    df_group1 = pd.concat([df_group1.reset_index(drop=True), embedding_df_group1.reset_index(drop=True)], axis=1)

    output_file = f'./src/data/{output_prefix}_group1.xlsx'
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_group1.to_excel(output_file, index=False)
    logger.info("✓ Group 1 embeddings saved to:", output_file=output_file)
    
    return output_file

def embed_group_2(dat_group2, output_prefix=""):
    
    #--------------------START----------------------- 
    # If dat_group2 is a list of dicts, convert to DataFrame
    if isinstance(dat_group2, list) and isinstance(dat_group2[0], dict):
            dat_group2 = pd.DataFrame(dat_group2)
    # Flatten hidden_state if it's shape (1, N)
    if isinstance(dat_group2, pd.DataFrame):
            dat_group2['hidden_state'] = dat_group2['hidden_state'].apply(
                lambda x: x.flatten() if hasattr(x, 'shape') and len(x.shape) == 2 and x.shape[0] == 1 else x
            )
    logger.info("\n=== Processing Group 2 (Dangerous Words) ===")
    logger.info("✓ Group 2 embeddings extracted - Shape:", shape=dat_group2['hidden_state'].shape)
    df_group2 = dat_group2 if isinstance(dat_group2, pd.DataFrame) else dat_group2.to_pandas()
    #---------------------END-----------------------
    
  # find out the size of each embedding vector
    hidden_size_group2 = df_group2['hidden_state'].iloc[0].shape[0]
    logger.info("Hidden size:", hidden_size=hidden_size_group2)

    #Expand the Embeddings into Separate Columns
    embedding_columns = [f"hidden_state_{i}" for i in range(hidden_size_group2)]

    # Expand the hidden_state arrays into separate columns
    embedding_values = np.vstack(df_group2['hidden_state'].values)
    embedding_df_group2 = pd.DataFrame(embedding_values, columns=embedding_columns)

    # Drop the original hidden_state column
    df_group2 = df_group2.drop(columns=['hidden_state'])

    # Concatenate the embeddings DataFrame with the original DataFrame
    df_group2 = pd.concat([df_group2.reset_index(drop=True), embedding_df_group2.reset_index(drop=True)], axis=1)

    output_file = f'./src/data/{output_prefix}_group2.xlsx'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_group2.to_excel(output_file, index=False)
    logger.info("✓ Group 2 embeddings saved to:", output_file=output_file)
    
    return output_file
