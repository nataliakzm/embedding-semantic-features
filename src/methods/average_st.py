import re
from sentence_transformers import SentenceTransformer, models
from src.utils import HF_TOKEN, logger

__all__ = [
    "extract_cls_pooling_st",
    "extract_max_pooling_st",
    "extract_mean_pooling_st",
    "extract_mean_sqrt_len_pooling_st",
    "extract_weightedmean_pooling_st",
    "extract_lasttoken_pooling_st"
    ]

def extract_cls_pooling_st(batch, model_name, device):
    """Use CLS token pooling."""
    if isinstance(batch, dict) and 'text' in batch:
        texts = batch['text']
    else:
        texts = batch
        
    word_emb = models.Transformer(model_name)
    
    if re.search(r"Llama", model_name):
        logger.warning("Llama tokenizer detected, setting pad token to eos token")
        if word_emb.tokenizer.pad_token is None:
            word_emb.tokenizer.pad_token = word_emb.tokenizer.eos_token
    
    pooling = models.Pooling(
        word_emb.get_word_embedding_dimension(), 
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False
        )
    model = SentenceTransformer(modules=[word_emb, pooling], device=device, use_auth_token=HF_TOKEN)
    embeddings = model.encode(texts)
    return {"hidden_state": embeddings}

def extract_max_pooling_st(batch, model_name, device):
    """
    Use max pooling.
    Takes the maximum value across all token embeddings for each embedding dimension. 
    """
    if isinstance(batch, dict) and 'text' in batch:
        texts = batch['text']
    else:
        texts = batch
        
    word_emb = models.Transformer(model_name)    
    
    if re.search(r"Llama", model_name):
        logger.warning("Llama tokenizer detected, setting pad token to eos token")
        if word_emb.tokenizer.pad_token is None:
            word_emb.tokenizer.pad_token = word_emb.tokenizer.eos_token
    
    pooling = models.Pooling(
        word_emb.get_word_embedding_dimension(), 
        pooling_mode_max_tokens=True,
        pooling_mode_mean_tokens=False
        )
    model = SentenceTransformer(modules=[word_emb, pooling], device=device, use_auth_token=HF_TOKEN)
    embeddings = model.encode(texts)
    return {"hidden_state": embeddings}

def extract_mean_pooling_st(batch, model_name, device):
    """
    Use mean pooling (default).
    Averages the embeddings of all tokens (excluding padding).
    """
    if isinstance(batch, dict) and 'text' in batch:
        texts = batch['text']
    else:
        texts = batch
        
    word_emb = models.Transformer(model_name)
    
    if re.search(r"Llama", model_name):
        logger.warning("Llama tokenizer detected, setting pad token to eos token")
        if word_emb.tokenizer.pad_token is None:
            word_emb.tokenizer.pad_token = word_emb.tokenizer.eos_token    
    
    pooling = models.Pooling(
        word_emb.get_word_embedding_dimension(), 
        pooling_mode_mean_tokens=True,
        )
    model = SentenceTransformer(modules=[word_emb, pooling], device=device, use_auth_token=HF_TOKEN)
    embeddings = model.encode(texts)
    return {"hidden_state": embeddings}

def extract_mean_sqrt_len_pooling_st(batch, model_name, device):
    """
    Use mean pooling divided by sqrt(length).
    Averages the embeddings of all tokens, then divides by the square root of the sentence length. This can help normalize for sentence length
    """
    if isinstance(batch, dict) and 'text' in batch:
        texts = batch['text']
    else:
        texts = batch
        
    word_emb = models.Transformer(model_name) 
    
    if re.search(r"Llama", model_name):
        logger.warning("Llama tokenizer detected, setting pad token to eos token")
        if word_emb.tokenizer.pad_token is None:
            word_emb.tokenizer.pad_token = word_emb.tokenizer.eos_token
    
    pooling = models.Pooling(
        word_emb.get_word_embedding_dimension(), 
        pooling_mode_mean_sqrt_len_tokens=True,
        pooling_mode_mean_tokens=False
        )
    model = SentenceTransformer(modules=[word_emb, pooling], device=device, use_auth_token=HF_TOKEN)
    embeddings = model.encode(texts)
    return {"hidden_state": embeddings}

def extract_weightedmean_pooling_st(batch, model_name, device):
    """
    Use weighted mean pooling.
    Computes a weighted mean of the token embeddings, where weights are typically based on token position or other criteria
    """
    if isinstance(batch, dict) and 'text' in batch:
        texts = batch['text']
    else:
        texts = batch
        
    word_emb = models.Transformer(model_name)    
    
    if re.search(r"Llama", model_name):
        logger.warning("Llama tokenizer detected, setting pad token to eos token")
        if word_emb.tokenizer.pad_token is None:
            word_emb.tokenizer.pad_token = word_emb.tokenizer.eos_token
    
    pooling = models.Pooling(
        word_emb.get_word_embedding_dimension(), 
        pooling_mode_weightedmean_tokens=True,
        pooling_mode_mean_tokens=False
        )
    model = SentenceTransformer(modules=[word_emb, pooling], device=device, use_auth_token=HF_TOKEN)
    embeddings = model.encode(texts)
    return {"hidden_state": embeddings}

def extract_lasttoken_pooling_st(batch, model_name, device):
    """
    Use last token pooling.
    Uses the embedding of the last token in the sequence as the sentence embedding
    """
    if isinstance(batch, dict) and 'text' in batch:
        texts = batch['text']
    else:
        texts = batch
        
    word_emb = models.Transformer(model_name)    
    
    if re.search(r"Llama", model_name):
        logger.warning("Llama tokenizer detected, setting pad token to eos token")
        if word_emb.tokenizer.pad_token is None:
            word_emb.tokenizer.pad_token = word_emb.tokenizer.eos_token
    
    pooling = models.Pooling(
        word_emb.get_word_embedding_dimension(), 
        pooling_mode_lasttoken=True,
        pooling_mode_mean_tokens=False
        )
    model = SentenceTransformer(modules=[word_emb, pooling], device=device, use_auth_token=HF_TOKEN)
    embeddings = model.encode(texts)
    return {"hidden_state": embeddings}