import re, traceback
from sentence_transformers import SentenceTransformer, models
from src import logger

def load_sentence_transformer_model(model_name, device, pooling_mode='mean'):
    """
    Load a sentence transformer model once

    Args:
        model_name: Name of the sentence transformer model
        device: torch.device to load model on
        pooling_mode: 'mean' for mean pooling, 'lasttoken' for last token pooling

    Returns:
        model: SentenceTransformer model or None if failed
    """
    try:
        device_str = str(device)
        logger.info("model_loading", model=model_name, pooling=pooling_mode, device=device_str)

        # Use last-token pooling for decoder-only models (Llama, DeepSeek, etc.)
        if pooling_mode == 'lasttoken':
            logger.info("building_lasttoken_pooling_model")

            # Load transformer - try different kwargs approaches for compatibility
            word_emb = None
            for kwargs in [
                {"model_kwargs": {"trust_remote_code": True}},
                {"backend_kwargs": {"trust_remote_code": True}},
                {"trust_remote_code": True},
                {},
            ]:
                try:
                    word_emb = models.Transformer(model_name, **kwargs)
                    break
                except TypeError:
                    continue

            if word_emb is None:
                raise RuntimeError(f"Failed to load transformer model: {model_name}")

            # Handle Llama tokenizer padding
            if re.search(r"Llama|llama", model_name):
                logger.warning("llama_tokenizer_detected", action="setting pad_token to eos_token")
                if word_emb.tokenizer.pad_token is None:
                    word_emb.tokenizer.pad_token = word_emb.tokenizer.eos_token

            # Create last-token pooling layer
            pooling = models.Pooling(
                word_emb.get_word_embedding_dimension(),
                pooling_mode_lasttoken=True,
                pooling_mode_mean_tokens=False
            )

            # Combine into SentenceTransformer
            try:
                model = SentenceTransformer(
                    modules=[word_emb, pooling],
                    device=device_str
                )
            except Exception as e:
                logger.warning("model_load_fallback", error=str(e), fallback="CPU")
                model = SentenceTransformer(
                    modules=[word_emb, pooling],
                    device='cpu'
                )
        else:
            # Default: use standard SentenceTransformer (mean pooling)
            if any(name in model_name for name in ["Qwen", "DeepSeek", "Llama"]):
                try:
                    model = SentenceTransformer(model_name, device=device_str, trust_remote_code=True)
                except Exception as e:
                    logger.warning("model_load_fallback", error=str(e), fallback="CPU")
                    model = SentenceTransformer(model_name, device='cpu', trust_remote_code=True)
            else:
                model = SentenceTransformer(model_name, device=device_str)

        logger.info("model_loaded_successfully", device=str(model.device))
        return model
    except Exception as e:
        logger.error("model_load_failed", model=model_name, error=str(e))
        traceback.print_exc()
        return None


def encode_with_model(model, sentences, show_progress=True):
    """
    Encode sentences using an already-loaded model

    Args:
        model: Loaded SentenceTransformer model
        sentences: List of sentences to encode
        show_progress: Whether to show progress bar

    Returns:
        embeddings: numpy array of shape (n_sentences, embedding_dim)
    """
    try:
        embeddings = model.encode(sentences, show_progress_bar=show_progress, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        logger.error("encoding_failed", error=str(e))
        return None

