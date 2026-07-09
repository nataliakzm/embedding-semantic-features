"""
Multi-Layer Neural Network for Common Sense Classification
Compares explicit representation (single-layer) vs learned representation (multi-layer)

Architecture: 4096 -> 500 -> 250 -> 1
- Uses same embeddings from config model (Qwen3-Embedding-8B)
- Hidden layers learn compressed representations
- Evaluates if learned features outperform explicit semantic axes
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from scipy import stats as scipy_stats

import argparse
import csv
import yaml
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from data.loader import load_dataset

from src import logger
from src.notify import send_slack_notification
from src.n_networks.multi_layer_nn import MultiLayerNN
from src.configure import load_config, get_config_value
from src.embd_model import load_sentence_transformer_model, encode_with_model
from src.save_embds import save_embeddings_to_excel
from src.analysis.compare_welights import compare_weights_with_axis, compare_rank_order
from src.analysis.scoring import get_feature_vector, compute_separation_metrics, axis_classification_accuracy
from src.analysis.ridge_probe import run_ridge_probe
from src.analysis.pca_analysis import run_pca_analysis


def train_multi_layer_nn(
    model, train_loader, val_loader,
    device, epochs=100, learning_rate=0.001,
    verbose_every=10, patience=50
):
    """
    Train multi-layer neural network using PyTorch with early stopping

    Args:
        model: MultiLayerNN instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: torch device
        epochs: Number of training epochs
        learning_rate: Learning rate
        verbose_every: Print progress every N epochs
        patience: Number of epochs to wait for improvement before stopping

    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Early stopping variables
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass (raw logits for numerically stable loss)
            optimizer.zero_grad()
            logits = model(inputs, return_logits=True)
            loss = criterion(logits.squeeze(), labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item()
            predictions = (logits.squeeze() >= 0.0).float()  # logit >= 0 ↔ prob >= 0.5
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                logits = model(inputs, return_logits=True)
                loss = criterion(logits.squeeze(), labels)

                val_loss += loss.item()
                predictions = (logits.squeeze() >= 0.0).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Early stopping check (>= so plateaus reset the patience counter)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Logging
        if verbose_every > 0 and (epoch + 1) % verbose_every == 0:
            logger.info("training_progress",
                       component="MultiLayerNN",
                       epoch=epoch + 1,
                       total_epochs=epochs,
                       train_loss=round(train_loss, 4),
                       train_acc=round(train_acc, 4),
                       val_loss=round(val_loss, 4),
                       val_acc=round(val_acc, 4),
                       best_val_acc=round(best_val_acc, 4),
                       epochs_no_improve=epochs_without_improvement)

        # Early stopping
        if patience > 0 and epochs_without_improvement >= patience:
            logger.info("early_stopping",
                       component="MultiLayerNN",
                       stopped_at_epoch=epoch + 1,
                       best_epoch=best_epoch,
                       best_val_acc=round(best_val_acc, 4),
                       patience=patience)
            break

    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("restored_best_model",
                   best_epoch=best_epoch,
                   best_val_acc=round(best_val_acc, 4))

    history['best_epoch'] = best_epoch
    history['best_val_acc'] = best_val_acc

    return history


def evaluate_model(model, data_loader, device):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            predictions = (outputs.squeeze() >= 0.5).float()
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    return accuracy, np.array(all_predictions), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(
        description='Train multi-layer NN for common sense classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_multi_layer_nn.py --config config.yaml
  python train_multi_layer_nn.py --config config.yaml --hidden1 500 --hidden2 250
        """
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file (required)')
    parser.add_argument('--hidden1', type=int, default=None,
                        help='Hidden layer 1 dimension (overrides config)')
    parser.add_argument('--hidden2', type=int, default=None,
                        help='Hidden layer 2 dimension (overrides config)')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate (overrides config)')
    args = parser.parse_args()

    logger.info("starting", app="Multi-Layer NN for Common Sense Classification")

    # Load config
    config = load_config(args.config)
    if config is None:
        return

    # Get parameters from config
    model_name = get_config_value(config, 'model', 'name')
    if not model_name:
        logger.error("config_missing_field", field="model.name")
        return

    pooling = get_config_value(config, 'model', 'pooling', default='mean')
    device_config = get_config_value(config, 'device', default='auto')
    if device_config == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    normalize = get_config_value(config, 'data', 'normalize_embeddings', default=False)

    # Multi-layer specific parameters (CLI overrides config)
    ml_learning_rate = get_config_value(config, 'multi_layer', 'learning_rate', default=0.001)
    ml_epochs = get_config_value(config, 'multi_layer', 'epochs', default=100)
    ml_batch_size = get_config_value(config, 'multi_layer', 'batch_size', default=64)
    verbose_every = get_config_value(config, 'multi_layer', 'verbose_every', default=10)
    early_stopping_patience = get_config_value(config, 'multi_layer', 'patience', default=50)

    # hidden1, hidden2, dropout: CLI args override config values
    args.hidden1 = args.hidden1 if args.hidden1 is not None else get_config_value(config, 'multi_layer', 'hidden1_dim', default=500)
    args.hidden2 = args.hidden2 if args.hidden2 is not None else get_config_value(config, 'multi_layer', 'hidden2_dim', default=250)
    args.dropout = args.dropout if args.dropout is not None else get_config_value(config, 'multi_layer', 'dropout', default=0.3)
    
    random_seed = get_config_value(config, 'data', 'random_seed', default=42)
    output_dir = get_config_value(config, 'output', 'multi_layer_dir', default='nn_results_multilayer')
    single_layer_dir = get_config_value(config, 'output', 'dir', default='nn_results')

    # Set seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    logger.info("configuration",
                model=model_name,
                pooling=pooling,
                device=str(device),
                normalize=normalize,
                hidden1_dim=args.hidden1,
                hidden2_dim=args.hidden2,
                dropout=args.dropout,
                learning_rate=ml_learning_rate,
                epochs=ml_epochs,
                batch_size=ml_batch_size,
                early_stopping_patience=early_stopping_patience)

    # Load data from config
    dataset_name = get_config_value(config, 'data', 'dataset', default='cs')
    dataset = load_dataset(dataset_name)
    csense0_train = dataset["train_label0"]
    csense1_train = dataset["train_label1"]
    csense0_test = dataset["test_label0"]
    csense1_test = dataset["test_label1"]

    logger.info("loading_data",
                dataset=dataset_name,
                train_class0=len(csense0_train),
                train_class1=len(csense1_train),
                test_class0=len(csense0_test),
                test_class1=len(csense1_test))

    train_sentences = csense0_train + csense1_train
    train_labels = np.array([0] * len(csense0_train) + [1] * len(csense1_train), dtype=np.float32)

    test_sentences = csense0_test + csense1_test
    test_labels = np.array([0] * len(csense0_test) + [1] * len(csense1_test), dtype=np.float32)

    shuffle_data = get_config_value(config, 'data', 'shuffle', default=False)
    if shuffle_data:
        rng = np.random.default_rng(random_seed)
        train_perm = rng.permutation(len(train_sentences))
        train_sentences = [train_sentences[i] for i in train_perm]
        train_labels = train_labels[train_perm]
        test_perm = rng.permutation(len(test_sentences))
        test_sentences = [test_sentences[i] for i in test_perm]
        test_labels = test_labels[test_perm]
        logger.info("data_shuffled", seed=random_seed,
                    train_size=len(train_sentences), test_size=len(test_sentences))

    # Read semantic axes config early so we can encode axis words while model is loaded
    semantic_axes = get_config_value(config, 'semantic_axes', default=None)
    axis_embeddings = {}  # Will cache {axis_name: (group1_emb, group2_emb)}

    # Check if embeddings already exist (to avoid regeneration)
    output_path = Path(output_dir)
    train_emb_file = output_path / "train_embeddings.npy"
    test_emb_file = output_path / "test_embeddings.npy"
    
    # Check multiple locations for cached embeddings:
    # 1. Multi-layer output directory
    # 2. Single-layer output directory  
    # 3. results/ folder (shared location)
    if train_emb_file.exists() and test_emb_file.exists():
        logger.info("loading_cached_embeddings", source="multi_layer_dir")
        train_embeddings = np.load(train_emb_file)
        test_embeddings = np.load(test_emb_file)
        logger.info("cached_embeddings_loaded",
                   train_shape=train_embeddings.shape,
                   test_shape=test_embeddings.shape)
    else:
        # Try loading from single-layer directory
        single_layer_path = Path(single_layer_dir)
        single_train_emb = single_layer_path / "train_embeddings.npy"
        single_test_emb = single_layer_path / "test_embeddings.npy"
        
        # Also check results/ folder
        results_path = Path("results")
        results_train_emb = results_path / "train_embeddings.npy"
        results_test_emb = results_path / "test_embeddings.npy"
        
        if single_train_emb.exists() and single_test_emb.exists():
            logger.info("loading_cached_embeddings", source="single_layer_dir")
            train_embeddings = np.load(single_train_emb)
            test_embeddings = np.load(single_test_emb)
            logger.info("cached_embeddings_loaded",
                       train_shape=train_embeddings.shape,
                       test_shape=test_embeddings.shape)
            
            # Copy to multi-layer directory for future use
            output_path.mkdir(exist_ok=True)
            np.save(train_emb_file, train_embeddings)
            np.save(test_emb_file, test_embeddings)
            logger.info("embeddings_copied_to_multi_layer_dir")
            
        elif results_train_emb.exists() and results_test_emb.exists():
            logger.info("loading_cached_embeddings", source="results_folder")
            train_embeddings = np.load(results_train_emb)
            test_embeddings = np.load(results_test_emb)
            logger.info("cached_embeddings_loaded",
                       train_shape=train_embeddings.shape,
                       test_shape=test_embeddings.shape)
            
            # Copy to multi-layer directory for future use
            output_path.mkdir(exist_ok=True)
            np.save(train_emb_file, train_embeddings)
            np.save(test_emb_file, test_embeddings)
            logger.info("embeddings_copied_to_multi_layer_dir")

        else:
            # Generate embeddings (neither directory has them)
            logger.info("generating_embeddings")
            model_st = load_sentence_transformer_model(model_name, device=device, pooling_mode=pooling)
            
            if model_st is None:
                logger.error("model_load_failed")
                return
            
            logger.info("encoding_train_sentences", count=len(train_sentences))
            train_embeddings = encode_with_model(model_st, train_sentences, show_progress=True)
            
            logger.info("encoding_test_sentences", count=len(test_sentences))
            test_embeddings = encode_with_model(model_st, test_sentences, show_progress=True)
            
            if train_embeddings is None or test_embeddings is None:
                logger.error("embedding_generation_failed")
                return
            
            # Encode semantic axis words while model is still loaded
            if semantic_axes:
                logger.info("encoding_semantic_axis_words", num_axes=len(semantic_axes))
                for axis_config in semantic_axes:
                    axis_name = axis_config.get('name', 'unnamed')
                    g1_emb = encode_with_model(model_st, axis_config.get('group1', []), show_progress=False)
                    g2_emb = encode_with_model(model_st, axis_config.get('group2', []), show_progress=False)
                    if g1_emb is not None and g2_emb is not None:
                        axis_embeddings[axis_name] = (g1_emb, g2_emb)

            # Free GPU memory immediately after generating embeddings
            del model_st
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("freed_gpu_memory", action="deleted embedding model after generation")

    # If axis embeddings weren't encoded yet (cached embeddings path), load model briefly
    if semantic_axes and not axis_embeddings:
        logger.info("loading_model_for_axis_words")
        model_st = load_sentence_transformer_model(model_name, device=device, pooling_mode=pooling)
        if model_st is not None:
            for axis_config in semantic_axes:
                axis_name = axis_config.get('name', 'unnamed')
                g1_emb = encode_with_model(model_st, axis_config.get('group1', []), show_progress=False)
                g2_emb = encode_with_model(model_st, axis_config.get('group2', []), show_progress=False)
                if g1_emb is not None and g2_emb is not None:
                    axis_embeddings[axis_name] = (g1_emb, g2_emb)
            del model_st
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("freed_gpu_memory", action="deleted embedding model after axis encoding")
        else:
            logger.warning("cannot_load_model_for_axes", reason="skipping semantic axis comparison")

    # Normalize if needed
    if normalize:
        logger.info("normalizing_embeddings")
        train_embeddings = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
        test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)
        # Normalize cached axis embeddings consistently
        for ax_name, (g1, g2) in axis_embeddings.items():
            axis_embeddings[ax_name] = (
                g1 / np.linalg.norm(g1, axis=1, keepdims=True),
                g2 / np.linalg.norm(g2, axis=1, keepdims=True),
            )
        logger.info("embeddings_normalized")
    
    # Save embeddings AFTER normalization (so cached embeddings are normalized)
    output_path.mkdir(exist_ok=True)
    np.save(train_emb_file, train_embeddings)
    np.save(test_emb_file, test_embeddings)
    logger.info("embeddings_saved", normalized=normalize)

    # Shuffle training data
    train_indices = np.random.permutation(len(train_embeddings))
    train_embeddings = train_embeddings[train_indices]
    train_labels = train_labels[train_indices]

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(train_embeddings)
    y_train = torch.FloatTensor(train_labels)
    X_test = torch.FloatTensor(test_embeddings)
    y_test = torch.FloatTensor(test_labels)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=ml_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=ml_batch_size, shuffle=False)

    logger.info("data_prepared",
                train_samples=len(X_train),
                test_samples=len(X_test),
                input_dim=train_embeddings.shape[1])

    # Initialize multi-layer network
    input_dim = train_embeddings.shape[1]
    model = MultiLayerNN(
        input_dim=input_dim,
        hidden1_dim=args.hidden1,
        hidden2_dim=args.hidden2,
        dropout=args.dropout
    )

    logger.info("model_architecture",
                layers=f"{input_dim} -> {args.hidden1} -> {args.hidden2} -> 1",
                total_params=sum(p.numel() for p in model.parameters()),
                trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Train the model
    logger.info("training_started")
    try:
        history = train_multi_layer_nn(
            model, train_loader, test_loader,
            device, epochs=ml_epochs,
            learning_rate=ml_learning_rate,
            verbose_every=verbose_every,
            patience=early_stopping_patience
        )
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            logger.warning("gpu_out_of_memory", 
                          error=str(e)[:100],
                          action="falling back to CPU")
            # Clear GPU and retry on CPU
            del model
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            
            # Recreate model on CPU
            model = MultiLayerNN(
                input_dim=input_dim,
                hidden1_dim=args.hidden1,
                hidden2_dim=args.hidden2,
                dropout=args.dropout
            )
            
            logger.info("retrying_on_cpu", device="cpu")
            history = train_multi_layer_nn(
                model, train_loader, test_loader,
                device, epochs=ml_epochs,
                learning_rate=ml_learning_rate,
                verbose_every=verbose_every,
                patience=early_stopping_patience
            )
        else:
            raise

    # Final evaluation
    logger.info("evaluating_model")
    train_acc, train_preds, _ = evaluate_model(model, train_loader, device)
    test_acc, test_preds, _ = evaluate_model(model, test_loader, device)

    logger.info("evaluation_complete",
                train_accuracy=round(train_acc, 4),
                test_accuracy=round(test_acc, 4))

    # Extract hidden representations
    logger.info("extracting_hidden_representations")
    model.eval()
    with torch.no_grad():
        test_hidden1 = model.get_hidden_representation(X_test.to(device), layer=1).cpu().numpy()
        test_hidden2 = model.get_hidden_representation(X_test.to(device), layer=2).cpu().numpy()
        train_hidden2 = model.get_hidden_representation(X_train.to(device), layer=2).cpu().numpy()

    logger.info("hidden_representations_extracted",
                hidden1_shape=test_hidden1.shape,
                hidden2_shape=test_hidden2.shape)

    # Extract output layer weights for semantic axis comparison
    logger.info("extracting_output_layer_weights")
    output_weights = model.fc3.weight.data.cpu().numpy().flatten()  # Shape: (250,)
    output_bias = model.fc3.bias.data.cpu().numpy()[0]
    
    logger.info("output_weights_extracted",
                shape=output_weights.shape,
                mean=round(float(np.mean(output_weights)), 4),
                std=round(float(np.std(output_weights)), 4))
    
    # Evaluate semantic axes using pre-cached axis embeddings
    if not semantic_axes:
        logger.error("config_missing_field", field="semantic_axes")
        return

    logger.info("evaluating_semantic_axes", num_axes=len(semantic_axes))

    # Free GPU memory by moving model to CPU temporarily
    model.to('cpu')
    model_device = 'cpu'

    axis_vectors_4096 = {}  # Collect for ridge probe / PCA
    axis_vectors_250 = {}   # Collect for ridge probe in hidden space

    # Compute NN separation metrics once (same for all axes)
    y_test_np = y_test.numpy()
    nn_separation = compute_separation_metrics(test_hidden2, y_test_np, output_weights, bias=output_bias)
    logger.info("nn_separation_metrics",
                geometric_separability_index=round(nn_separation['geometric_separability_index'], 4),
                effect_size=round(nn_separation['mann_whitney_u']['effect_size'], 4),
                mean_class0=round(nn_separation['mean_score_class0'], 4),
                mean_class1=round(nn_separation['mean_score_class1'], 4),
                mean_difference=round(nn_separation['mean_difference'], 4))

    if not axis_embeddings:
        logger.warning("cannot_load_model_for_axes", reason="skipping semantic axis comparison")
        semantic_axes_results = {}
    else:
        semantic_axes_results = {}

        for axis_config in semantic_axes:
            axis_name = axis_config.get('name', 'unnamed')
            group1_words = axis_config.get('group1', [])
            group2_words = axis_config.get('group2', [])

            if axis_name not in axis_embeddings:
                logger.error("semantic_axis_embedding_failed", axis_name=axis_name)
                continue

            group1_emb, group2_emb = axis_embeddings[axis_name]

            logger.info("computing_semantic_axis",
                       axis_name=axis_name,
                       group1=group1_words,
                       group2=group2_words)

            # Get axis in 4096-dim space
            group1_list = [group1_emb[i] for i in range(len(group1_emb))]
            group2_list = [group2_emb[i] for i in range(len(group2_emb))]
            axis_vector_4096 = get_feature_vector(group1_list, group2_list)
            axis_vectors_4096[axis_name] = axis_vector_4096

            # Project axis to hidden2 space (250-dim) using hidden layer weights
            # axis_vector_250 = axis_vector_4096 @ W1 @ W2 (where W1, W2 are weight matrices)
            # For simplicity, we'll project test embeddings and compute axis in 250-dim space
            
            with torch.no_grad():
                group1_tensor = torch.FloatTensor(group1_emb).to(model_device)
                group2_tensor = torch.FloatTensor(group2_emb).to(model_device)
                
                # Get hidden2 representations for axis words
                group1_hidden2 = model.get_hidden_representation(group1_tensor, layer=2).cpu().numpy()
                group2_hidden2 = model.get_hidden_representation(group2_tensor, layer=2).cpu().numpy()
            
            # Compute axis in 250-dim hidden space
            group1_h2_list = [group1_hidden2[i] for i in range(len(group1_hidden2))]
            group2_h2_list = [group2_hidden2[i] for i in range(len(group2_hidden2))]
            axis_vector_250 = get_feature_vector(group1_h2_list, group2_h2_list)
            axis_vectors_250[axis_name] = axis_vector_250

            # Compare output weights (250-dim) with axis (250-dim)
            comparison = compare_weights_with_axis(output_weights, axis_vector_250)
            
            spearman_corr, spearman_pval = scipy_stats.spearmanr(output_weights, axis_vector_250)
            
            logger.info("axis_comparison_results",
                       axis_name=axis_name,
                       cosine_similarity=round(comparison['cosine_similarity'], 4),
                       angle_degrees=round(comparison['angle_degrees'], 2),
                       correlation=round(comparison['correlation'], 4),
                       spearman_correlation=round(spearman_corr, 4),
                       spearman_pvalue=f"{spearman_pval:.2e}")
            
            # Compute separation metrics using hidden2 space
            axis_separation = compute_separation_metrics(test_hidden2, y_test_np, axis_vector_250)

            nn_gsi = nn_separation['geometric_separability_index']
            nn_scores = np.dot(test_hidden2, output_weights) + output_bias

            gsi_advantage = nn_gsi - axis_separation['geometric_separability_index']
            
            logger.info("axis_separation_metrics",
                       axis_name=axis_name,
                       nn_gsi=round(nn_gsi, 4),
                       axis_gsi=round(axis_separation['geometric_separability_index'], 4),
                       nn_gsi_advantage=round(gsi_advantage, 4))

            # Axis classification accuracy in 250-dim hidden space
            axis_cls_250 = axis_classification_accuracy(
                train_hidden2, train_labels, test_hidden2, y_test_np, axis_vector_250)
            logger.info("axis_classification_accuracy",
                       axis_name=axis_name,
                       space="250-dim",
                       train_accuracy=round(axis_cls_250['train_accuracy'], 4),
                       test_accuracy=round(axis_cls_250['test_accuracy'], 4))

            # Axis classification accuracy in 4096-dim original space
            axis_cls_4096 = axis_classification_accuracy(
                train_embeddings, train_labels, test_embeddings, test_labels, axis_vector_4096)
            logger.info("axis_classification_accuracy",
                       axis_name=axis_name,
                       space="4096-dim",
                       train_accuracy=round(axis_cls_4096['train_accuracy'], 4),
                       test_accuracy=round(axis_cls_4096['test_accuracy'], 4))

            # Rank-order comparison
            axis_dot_250 = np.dot(axis_vector_250, axis_vector_250)
            axis_scores_250 = np.dot(test_hidden2, axis_vector_250) / axis_dot_250
            rank_comparison = compare_rank_order(nn_scores, axis_scores_250, labels=y_test_np)

            logger.info("rank_order_comparison",
                       axis_name=axis_name,
                       spearman_correlation=round(rank_comparison['spearman_correlation'], 4),
                       spearman_pvalue=f"{rank_comparison['spearman_pvalue']:.2e}",
                       mean_displacement=round(rank_comparison['mean_displacement'], 2),
                       median_displacement=round(rank_comparison['median_displacement'], 2),
                       max_displacement=rank_comparison['max_displacement'],
                       matrix_correlation=round(rank_comparison['matrix_correlation'], 4))

            # Save rank comparison outputs
            output_path.mkdir(exist_ok=True)
            np.save(output_path / f"rank_displacements_{axis_name}.npy", rank_comparison['rank_displacements'])

            csv_path = output_path / f"rank_comparison_{axis_name}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_id', 'label', 'nn_score', 'axis_score', 'nn_rank', 'axis_rank', 'displacement'])
                for i in range(len(nn_scores)):
                    writer.writerow([
                        i, int(y_test_np[i]),
                        round(float(nn_scores[i]), 6), round(float(axis_scores_250[i]), 6),
                        int(rank_comparison['nn_ranks'][i]), int(rank_comparison['axis_ranks'][i]),
                        int(rank_comparison['rank_displacements'][i])
                    ])

            # Store results
            semantic_axes_results[axis_name] = {
                'group1_words': group1_words,
                'group2_words': group2_words,
                'comparison': {
                    'cosine_similarity': float(comparison['cosine_similarity']),
                    'angle_degrees': float(comparison['angle_degrees']),
                    'correlation': float(comparison['correlation']),
                    'spearman_correlation': float(spearman_corr),
                    'spearman_pvalue': float(spearman_pval)
                },
                'rank_order': {
                    'spearman_correlation': float(rank_comparison['spearman_correlation']),
                    'spearman_pvalue': float(rank_comparison['spearman_pvalue']),
                    'mean_displacement': float(rank_comparison['mean_displacement']),
                    'median_displacement': float(rank_comparison['median_displacement']),
                    'max_displacement': int(rank_comparison['max_displacement']),
                    'matrix_correlation': float(rank_comparison['matrix_correlation']),
                    'mean_displacement_class0': float(rank_comparison.get('mean_displacement_class0', 0)),
                    'mean_displacement_class1': float(rank_comparison.get('mean_displacement_class1', 0)),
                },
                'nn_gsi': float(nn_gsi),
                'axis_gsi': float(axis_separation['geometric_separability_index']),
                'nn_gsi_advantage': float(gsi_advantage),
                'axis_classification': {
                    '250_dim': {
                        'train_accuracy': axis_cls_250['train_accuracy'],
                        'test_accuracy': axis_cls_250['test_accuracy'],
                        'threshold': axis_cls_250['threshold'],
                        'direction': axis_cls_250['direction'],
                    },
                    '4096_dim': {
                        'train_accuracy': axis_cls_4096['train_accuracy'],
                        'test_accuracy': axis_cls_4096['test_accuracy'],
                        'threshold': axis_cls_4096['threshold'],
                        'direction': axis_cls_4096['direction'],
                    },
                }
            }

    # Ridge regression probe in 4096-dim space (Gurnee & Tegmark, 2024)
    ridge_enabled = get_config_value(config, 'ridge_probe', 'enabled', default=False)
    ridge_results = {}
    if ridge_enabled and axis_vectors_4096:
        ridge_alpha = get_config_value(config, 'ridge_probe', 'alpha', default=1.0)
        # Use a simple linear weight vector for comparison (output_weights is 250-dim, not comparable)
        # So we run ridge in 4096-dim space and compare against 4096-dim axis vectors
        ridge_results, ridge_weights_vec = run_ridge_probe(
            train_embeddings, train_labels, test_embeddings, test_labels,
            np.zeros(train_embeddings.shape[1]),  # No single NN weight in 4096-dim for ML
            0.0, axis_vectors_4096, alpha=ridge_alpha)
        output_path.mkdir(exist_ok=True)
        np.save(output_path / "ridge_weights.npy", ridge_weights_vec)

    # Ridge regression probe in 250-dim hidden space
    ridge_results_250 = {}
    if ridge_enabled and axis_vectors_250:
        ridge_alpha = get_config_value(config, 'ridge_probe', 'alpha', default=1.0)
        ridge_results_250, ridge_weights_250_vec = run_ridge_probe(
            train_hidden2, train_labels, test_hidden2, test_labels,
            output_weights, output_bias, axis_vectors_250, alpha=ridge_alpha)
        np.save(output_path / "ridge_weights_250.npy", ridge_weights_250_vec)
        ridge_results['hidden_space'] = ridge_results_250

    # PCA dimensionality analysis in 4096-dim space (Gurnee & Tegmark, 2024)
    pca_enabled = get_config_value(config, 'pca_analysis', 'enabled', default=False)
    pca_results = {}
    if pca_enabled and axis_vectors_4096:
        pca_k_values = get_config_value(config, 'pca_analysis', 'k_values',
                                         default=[50, 100, 250, 500, 1000, 2000])
        pca_results = run_pca_analysis(
            train_embeddings, train_labels, test_embeddings, test_labels,
            axis_vectors_4096, k_values=pca_k_values,
            training_params={
                'learning_rate': 0.01,
                'epochs': ml_epochs // 2,
                'batch_size': ml_batch_size,
                'patience': early_stopping_patience // 2 if early_stopping_patience else 0,
                'l2_lambda': 0,
            })

    # Save results
    output_path.mkdir(exist_ok=True)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden1_dim': args.hidden1,
        'hidden2_dim': args.hidden2,
        'dropout': args.dropout,
    }, output_path / "multi_layer_model.pth")
    
    # Save output weights and bias separately (for GSI export compatibility)
    np.save(output_path / "nn_weights.npy", output_weights)
    np.save(output_path / "nn_bias.npy", np.array([output_bias]))
    
    # Save history
    np.save(output_path / "train_history.npy", history)
    
    # Save hidden representations
    np.save(output_path / "test_hidden1.npy", test_hidden1)
    np.save(output_path / "test_hidden2.npy", test_hidden2)
    
    # Save test labels for GSI export compatibility
    np.save(output_path / "test_labels.npy", y_test.numpy())
    
    # Save embeddings to Excel format
    save_embeddings = get_config_value(config, 'output', 'save_embeddings', default=False)
    if save_embeddings:
        logger.info("saving_embeddings_to_excel")
        # Load from .npy (already normalized if normalize=True)
        train_emb_for_xlsx = np.load(output_path / "train_embeddings.npy")
        test_emb_for_xlsx = np.load(output_path / "test_embeddings.npy")
        train_labels_orig = np.array([0] * len(csense0_train) + [1] * len(csense1_train))
        test_labels_orig = np.array([0] * len(csense0_test) + [1] * len(csense1_test))
        train_excel = save_embeddings_to_excel(
            train_emb_for_xlsx, train_sentences, 
            train_labels_orig, 
            output_path, prefix="train")
        test_excel = save_embeddings_to_excel(
            test_emb_for_xlsx, test_sentences, 
            test_labels_orig, output_path, prefix="test")
        logger.info("embeddings_saved_to_excel", 
                   train_file=str(train_excel), 
                   test_file=str(test_excel),
                   normalized=normalize)
    
    # Save config
    results_config = {
        'model_name': model_name,
        'pooling': pooling,
        'device': str(device),
        'normalize': normalize,
        'architecture': {
            'input_dim': input_dim,
            'hidden1_dim': args.hidden1,
            'hidden2_dim': args.hidden2,
            'dropout': args.dropout,
        },
        'training': {
            'learning_rate': ml_learning_rate,
            'epochs': ml_epochs,
            'batch_size': ml_batch_size,
            'early_stopping_patience': early_stopping_patience,
        },
        'results': {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
            'best_epoch': history.get('best_epoch', ml_epochs),
            'best_val_acc': history.get('best_val_acc', float(test_acc)),
            'actual_epochs': len(history['train_loss']),
        },
        'nn_separation_metrics': {
            'geometric_separability_index': float(nn_separation['geometric_separability_index']),
            'mann_whitney_p_value': float(nn_separation['mann_whitney_u']['p_value']),
            'mann_whitney_significant': nn_separation['mann_whitney_u']['significant'],
            'effect_size': float(nn_separation['mann_whitney_u']['effect_size']),
            'mean_class0': float(nn_separation['mean_score_class0']),
            'mean_class1': float(nn_separation['mean_score_class1']),
            'mean_difference': float(nn_separation['mean_difference']),
        },
        'semantic_axes': semantic_axes_results,
        'ridge_probe': ridge_results,
        'pca_analysis': pca_results
    }
    
    def _to_native(obj):
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_native(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    with open(output_path / "run_config.yaml", 'w') as f:
        yaml.safe_dump(_to_native(results_config), f, default_flow_style=False)
    
    saved_files = [
        "multi_layer_model.pth",
        "nn_weights.npy",
        "nn_bias.npy",
        "train_history.npy",
        "test_hidden1.npy",
        "test_hidden2.npy",
        "test_labels.npy",
        "run_config.yaml"
    ]
    if save_embeddings:
        saved_files.extend(["train_embeddings.xlsx", "test_embeddings.xlsx"])
    
    logger.info("results_saved",
                output_dir=str(output_path),
                files=saved_files)
    
    logger.info("training_complete")
    send_slack_notification(
        f"[Embd-NN] ML training complete — dataset={dataset_name}, "
        f"test_acc={test_acc:.4f}, output={output_path}"
    )

    # Comparison summary
    logger.info("comparison_summary",
                info="Multi-layer NN allows learning compressed representations",
                note="Compare test_accuracy with single-layer results to see improvement from learned features")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        send_slack_notification(f"[Embd-NN] ML training FAILED: {e}")
        raise
