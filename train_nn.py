"""
Single-layer Neural Network for Common Sense Classification
Trains on group7_2500 data and compares learned weights with semantic axis
"""
import torch
import csv
import yaml, argparse
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from data.loader import load_dataset

from src import logger
from src.notify import send_slack_notification
from src.n_networks.single_layer_nn import SingleLayerNN
from src.configure import load_config, get_config_value
from src.embd_model import load_sentence_transformer_model, encode_with_model
from src.save_embds import save_embeddings_to_excel
from src.analysis.compare_welights import compare_weights_with_axis, compare_rank_order
from src.analysis.scoring import get_feature_vector, compute_separation_metrics, axis_classification_accuracy
from src.analysis.evaluate_pairwise import evaluate_pairwise
from src.analysis.ridge_probe import run_ridge_probe
from src.analysis.pca_analysis import run_pca_analysis


def main():
    parser = argparse.ArgumentParser(
        description='Train single-layer NN on common sense data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples: python train_nn.py --config config.yaml"
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file (required)')
    args = parser.parse_args()

    logger.info("starting", app="Single-Layer NN")
    config = load_config(args.config)
    if config is None:
        return
    
    model_name = get_config_value(config, 'model', 'name')
    if not model_name:
        logger.error("config_missing_field", field="model.name")
        return
    
    pooling = get_config_value(config, 'model', 'pooling', default='mean')

    # Device selection from config or auto-detect
    device_config = config.get('device', 'auto')
    if device_config == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    
    normalize = get_config_value(config, 'data', 'normalize_embeddings', default=False)
    
    learning_rate = get_config_value(config, 'training', 'learning_rate', default=0.01)
    epochs = get_config_value(config, 'training', 'epochs', default=1000)
    batch_size = get_config_value(config, 'training', 'batch_size', default=32)
    verbose_every = get_config_value(config, 'training', 'verbose_every', default=100)
    early_stopping_patience = get_config_value(config, 'training', 'patience', default=0)
    l2_lambda = get_config_value(config, 'training', 'l2_lambda', default=0.0)

    random_seed = get_config_value(config, 'data', 'random_seed', default=42)
    
    output_dir = get_config_value(config, 'output', 'dir', default='nn_results')
    save_weights = get_config_value(config, 'output', 'save_weights', default=True)
    save_axis = get_config_value(config, 'output', 'save_axis', default=True)
    save_history = get_config_value(config, 'output', 'save_history', default=True)
    save_embeddings = get_config_value(config, 'output', 'save_embeddings', default=False)
    
    # Get semantic axes (list of axes to compare against)
    semantic_axes = get_config_value(config, 'semantic_axes', default=None)
    if not semantic_axes:
        logger.error("config_missing_field", field="semantic_axes")
        return
    
    logger.info("configuration",
                model=model_name,
                pooling=pooling,
                device=str(device),
                normalize=normalize,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                early_stopping_patience=early_stopping_patience,
                l2_lambda=l2_lambda)

    # Load dataset based on config
    dataset_name = get_config_value(config, 'data', 'dataset', default='cs')
    dataset = load_dataset(dataset_name)
    csense0_train = dataset["train_label0"]
    csense1_train = dataset["train_label1"]
    csense0_test = dataset["test_label0"]
    csense1_test = dataset["test_label1"]
    sentence_pairs = dataset["sentence_pairs"]

    logger.info("loading_training_data",
                dataset=dataset_name,
                class0_train=len(csense0_train),
                class1_train=len(csense1_train))

    logger.info("loading_test_data",
                class0_test=len(csense0_test),
                class1_test=len(csense1_test))

    train_sentences = csense0_train + csense1_train
    train_labels = np.array([0] * len(csense0_train) + [1] * len(csense1_train))

    test_sentences = csense0_test + csense1_test
    test_labels = np.array([0] * len(csense0_test) + [1] * len(csense1_test))

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

    axis_embeddings = {}  # Will cache {axis_name: (group1_emb, group2_emb)}

    # Check if embeddings already exist (to avoid regeneration)
    output_path = Path(output_dir)
    train_emb_file = output_path / "train_embeddings.npy"
    test_emb_file = output_path / "test_embeddings.npy"
    
    # Also check results/ folder
    results_path = Path("results")
    results_train_emb = results_path / "train_embeddings.npy"
    results_test_emb = results_path / "test_embeddings.npy"
    
    if train_emb_file.exists() and test_emb_file.exists():
        logger.info("loading_cached_embeddings", source="output_dir")
        train_embeddings = np.load(train_emb_file)
        test_embeddings = np.load(test_emb_file)
        logger.info("cached_embeddings_loaded",
                   train_shape=train_embeddings.shape,
                   test_shape=test_embeddings.shape)
        model = None  # Don't need to load model

    elif results_train_emb.exists() and results_test_emb.exists():
        logger.info("loading_cached_embeddings", source="results_folder")
        train_embeddings = np.load(results_train_emb)
        test_embeddings = np.load(results_test_emb)
        logger.info("cached_embeddings_loaded",
                   train_shape=train_embeddings.shape,
                   test_shape=test_embeddings.shape)
        model = None  # Don't need to load model

    else:
        # Generate embeddings
        model = load_sentence_transformer_model(model_name, device=device, pooling_mode=pooling)

        if model is None:
            logger.error("model_loading_failed", model=model_name)
            return

        logger.info("generating_train_embeddings", sentence_count=len(train_sentences))
        train_embeddings = encode_with_model(model, train_sentences, show_progress=True)
        if train_embeddings is None:
            logger.error("train_embedding_generation_failed")
            return

        logger.info("train_embeddings_generated", shape=train_embeddings.shape)

        logger.info("generating_test_embeddings", sentence_count=len(test_sentences))
        test_embeddings = encode_with_model(model, test_sentences, show_progress=True)
        if test_embeddings is None:
            logger.error("test_embedding_generation_failed")
            return

        logger.info("test_embeddings_generated", shape=test_embeddings.shape)

        # Encode semantic axis words while model is still loaded
        if semantic_axes:
            logger.info("encoding_semantic_axis_words", num_axes=len(semantic_axes))
            for axis_config in semantic_axes:
                ax_name = axis_config.get('name', 'unnamed')
                g1_emb = encode_with_model(model, axis_config.get('group1', []), show_progress=False)
                g2_emb = encode_with_model(model, axis_config.get('group2', []), show_progress=False)
                if g1_emb is not None and g2_emb is not None:
                    axis_embeddings[ax_name] = (g1_emb, g2_emb)

        # Free GPU memory after generating embeddings
        model = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("freed_gpu_memory", action="deleted embedding model after generation")

    # Optional: Normalize embeddings
    if normalize:
        logger.info("normalizing_embeddings")
        train_norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True)
        train_embeddings = train_embeddings / train_norms
        test_norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True)
        test_embeddings = test_embeddings / test_norms
        logger.info("embeddings_normalized")

    # Shuffle training data
    np.random.seed(random_seed)
    train_indices = np.random.permutation(len(train_embeddings))
    X_train = train_embeddings[train_indices]
    y_train = train_labels[train_indices]
    
    X_test = test_embeddings
    y_test = test_labels

    logger.info("data_loaded", 
                train_samples=len(X_train), 
                test_samples=len(X_test))

    # Train neural network
    logger.info("training_started",
                architecture=f"{train_embeddings.shape[1]} inputs -> 1 output",
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size)

    nn = SingleLayerNN(input_dim=train_embeddings.shape[1])
    nn.train(X_train, y_train,
             X_val=X_test, y_val=y_test,
             learning_rate=learning_rate,
             epochs=epochs,
             batch_size=batch_size,
             verbose=True,
             verbose_every=verbose_every,
             patience=early_stopping_patience,
             l2_lambda=l2_lambda)

    # Evaluate
    train_preds = nn.predict(X_train)
    test_preds = nn.predict(X_test)

    train_acc = np.mean(train_preds == y_train)
    test_acc = np.mean(test_preds == y_test)

    logger.info("evaluation_complete",
                train_accuracy=round(train_acc, 4),
                test_accuracy=round(test_acc, 4))

    # Get NN weights
    nn_weights = nn.get_weights()

    # Weight statistics
    logger.info("weight_statistics",
                nn_weights_mean=round(float(np.mean(nn_weights)), 4),
                nn_weights_std=round(float(np.std(nn_weights)), 4),
                nn_weights_min=round(float(np.min(nn_weights)), 4),
                nn_weights_max=round(float(np.max(nn_weights)), 4))

    # Compute separation metrics for NN weights (only once)
    # Include bias to match the NN's actual decision boundary
    logger.info("computing_nn_separation_metrics")
    nn_separation = compute_separation_metrics(X_test, y_test, nn_weights, bias=nn.bias)
    logger.info("nn_separation_metrics",
                geometric_separability_index=round(nn_separation['geometric_separability_index'], 4),
                mann_whitney_p_value=f"{nn_separation['mann_whitney_u']['p_value']:.2e}",
                mann_whitney_significant=nn_separation['mann_whitney_u']['significant'],
                effect_size=round(nn_separation['mann_whitney_u']['effect_size'], 4),
                mean_difference=round(nn_separation['mean_difference'], 4))

    # Evaluate against all semantic axes
    logger.info("evaluating_semantic_axes", num_axes=len(semantic_axes))

    # Load model for semantic axis embeddings
    logger.info("loading_model_for_semantic_axes")
    model = load_sentence_transformer_model(model_name, device=device, pooling_mode=pooling)
    if model is None:
        logger.error("model_loading_failed", model=model_name)
        return

    all_axes_results = {}
    axis_vectors = {}  # Store for pairwise evaluation

    for axis_config in semantic_axes:
        axis_name = axis_config.get('name', 'unnamed')
        group1_words = axis_config.get('group1', [])
        group2_words = axis_config.get('group2', [])

        logger.info("computing_semantic_axis",
                    axis_name=axis_name,
                    group1=group1_words,
                    group2=group2_words)

        group1_emb = encode_with_model(model, group1_words, show_progress=False)
        group2_emb = encode_with_model(model, group2_words, show_progress=False)

        if group1_emb is None or group2_emb is None:
            logger.error("semantic_axis_embedding_failed", axis_name=axis_name)
            continue

        group1_list = [group1_emb[i] for i in range(len(group1_emb))]
        group2_list = [group2_emb[i] for i in range(len(group2_emb))]

        axis_vector = get_feature_vector(group1_list, group2_list)
        axis_vectors[axis_name] = axis_vector

        # Compare weights with this axis
        comparison = compare_weights_with_axis(nn_weights, axis_vector)

        logger.info("axis_comparison_results",
                    axis_name=axis_name,
                    cosine_similarity=round(comparison['cosine_similarity'], 4),
                    angle_degrees=round(comparison['angle_degrees'], 2),
                    correlation=round(comparison['correlation'], 4),
                    axis_mean=round(float(np.mean(axis_vector)), 4),
                    axis_std=round(float(np.std(axis_vector)), 4))

        # Compute separation metrics for this axis
        axis_separation = compute_separation_metrics(X_test, y_test, axis_vector)

        gsi_advantage = nn_separation['geometric_separability_index'] - axis_separation['geometric_separability_index']

        logger.info("axis_separation_metrics",
                    axis_name=axis_name,
                    geometric_separability_index=round(axis_separation['geometric_separability_index'], 4),
                    mann_whitney_p_value=f"{axis_separation['mann_whitney_u']['p_value']:.2e}",
                    effect_size=round(axis_separation['mann_whitney_u']['effect_size'], 4),
                    nn_gsi_advantage=round(gsi_advantage, 4))

        # Axis classification accuracy (use axis as a binary classifier)
        axis_cls = axis_classification_accuracy(X_train, y_train, X_test, y_test, axis_vector)
        logger.info("axis_classification_accuracy",
                    axis_name=axis_name,
                    train_accuracy=round(axis_cls['train_accuracy'], 4),
                    test_accuracy=round(axis_cls['test_accuracy'], 4),
                    threshold=round(axis_cls['threshold'], 4))

        # Rank-order comparison
        nn_scores = np.dot(X_test, nn_weights) + nn.bias
        axis_dot = np.dot(axis_vector, axis_vector)
        axis_scores = np.dot(X_test, axis_vector) / axis_dot
        rank_comparison = compare_rank_order(nn_scores, axis_scores, labels=y_test)

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
                    i, int(y_test[i]),
                    round(float(nn_scores[i]), 6), round(float(axis_scores[i]), 6),
                    int(rank_comparison['nn_ranks'][i]), int(rank_comparison['axis_ranks'][i]),
                    int(rank_comparison['rank_displacements'][i])
                ])

        # Store results for this axis
        all_axes_results[axis_name] = {
            'group1_words': group1_words,
            'group2_words': group2_words,
            'comparison': {
                'cosine_similarity': float(comparison['cosine_similarity']),
                'angle_degrees': float(comparison['angle_degrees']),
                'correlation': float(comparison['correlation'])
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
            'separation': {
                'geometric_separability_index': float(axis_separation['geometric_separability_index']),
                'mann_whitney_p_value': float(axis_separation['mann_whitney_u']['p_value']),
                'mann_whitney_significant': axis_separation['mann_whitney_u']['significant'],
                'effect_size': float(axis_separation['mann_whitney_u']['effect_size']),
                'mean_class0': float(axis_separation['mean_score_class0']),
                'mean_class1': float(axis_separation['mean_score_class1']),
                'mean_difference': float(axis_separation['mean_difference'])
            },
            'axis_classification': {
                'train_accuracy': axis_cls['train_accuracy'],
                'test_accuracy': axis_cls['test_accuracy'],
                'threshold': axis_cls['threshold'],
                'direction': axis_cls['direction'],
            },
            'nn_gsi_advantage': float(gsi_advantage)
        }

    # Ridge regression probe (Gurnee & Tegmark, 2024)
    ridge_enabled = get_config_value(config, 'ridge_probe', 'enabled', default=False)
    ridge_results = {}
    if ridge_enabled and axis_vectors:
        ridge_alpha = get_config_value(config, 'ridge_probe', 'alpha', default=1.0)
        ridge_results, ridge_weights_vec = run_ridge_probe(
            X_train, y_train, X_test, y_test,
            nn_weights, nn.bias, axis_vectors, alpha=ridge_alpha)
        # Save ridge weights
        output_path.mkdir(exist_ok=True)
        np.save(output_path / "ridge_weights.npy", ridge_weights_vec)

    # PCA dimensionality analysis (Gurnee & Tegmark, 2024)
    pca_enabled = get_config_value(config, 'pca_analysis', 'enabled', default=False)
    pca_results = {}
    if pca_enabled and axis_vectors:
        pca_k_values = get_config_value(config, 'pca_analysis', 'k_values',
                                         default=[50, 100, 250, 500, 1000, 2000])
        pca_results = run_pca_analysis(
            X_train, y_train, X_test, y_test, axis_vectors,
            k_values=pca_k_values,
            training_params={
                'learning_rate': learning_rate,
                'epochs': epochs // 2,  # Faster for PCA sub-runs
                'batch_size': batch_size,
                'patience': early_stopping_patience // 2 if early_stopping_patience else 0,
                'l2_lambda': l2_lambda,
            })

    # Pairwise evaluation (requires comparison pairs, not available for all datasets)
    if sentence_pairs is None:
        logger.info("skipping_pairwise_evaluation", reason="no comparison pairs for this dataset")
        pairwise_results_all = {}
    else:
        logger.info("starting_pairwise_evaluation", total_pairs=len(sentence_pairs))

        # Build embeddings dictionary from all sentences in pairs
        # Preserve order while removing duplicates (same sentence = same embedding)
        all_pair_sentences = []
        seen_sentences = set()
        for non_cs, cs in sentence_pairs:
            non_cs_key = non_cs.lower().strip()
            cs_key = cs.lower().strip()
            if non_cs_key not in seen_sentences:
                all_pair_sentences.append(non_cs_key)
                seen_sentences.add(non_cs_key)
            if cs_key not in seen_sentences:
                all_pair_sentences.append(cs_key)
                seen_sentences.add(cs_key)

        unique_pair_sentences = all_pair_sentences
        logger.info("generating_pair_embeddings",
                    unique_sentences=len(unique_pair_sentences),
                    total_in_pairs=len(sentence_pairs) * 2)

        pair_embeddings = encode_with_model(model, unique_pair_sentences, show_progress=True)

        pairwise_results_all = {}

        if pair_embeddings is None:
            logger.error("pair_embedding_generation_failed")
        else:
            # Normalize if needed
            if normalize:
                pair_norms = np.linalg.norm(pair_embeddings, axis=1, keepdims=True)
                pair_embeddings = pair_embeddings / pair_norms

            # Create embeddings dictionary
            embeddings_dict = {s: pair_embeddings[i] for i, s in enumerate(unique_pair_sentences)}

            # Run pairwise evaluation for NN and all axes
            for axis_name, axis_vector in axis_vectors.items():
                pairwise_results = evaluate_pairwise(
                    sentence_pairs, embeddings_dict, nn_weights, axis_vector, nn.bias
                )

                logger.info("pairwise_results",
                            axis_name=axis_name,
                            nn_accuracy=f"{pairwise_results['nn_accuracy']:.2%}",
                            axis_accuracy=f"{pairwise_results['axis_accuracy']:.2%}",
                            nn_advantage=f"{pairwise_results['nn_advantage']:+.2%}")

                pairwise_results_all[axis_name] = {
                    'total_pairs': pairwise_results['total_pairs'],
                    'nn_correct': pairwise_results['nn_correct'],
                    'nn_accuracy': float(pairwise_results['nn_accuracy']),
                    'axis_correct': pairwise_results['axis_correct'],
                    'axis_accuracy': float(pairwise_results['axis_accuracy']),
                    'nn_advantage': float(pairwise_results['nn_advantage'])
                }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    saved_files = []
    
    if save_weights:
        np.save(output_path / "nn_weights.npy", nn_weights)
        np.save(output_path / "nn_bias.npy", np.array([nn.bias]))  # Save bias separately
        saved_files.extend(["nn_weights.npy", "nn_bias.npy"])
    
    if save_axis:
        for axis_name, axis_vector in axis_vectors.items():
            filename = f"axis_vector_{axis_name}.npy"
            np.save(output_path / filename, axis_vector)
            saved_files.append(filename)
    
    if save_history:
        np.save(output_path / "train_history.npy", nn.history)
        saved_files.append("train_history.npy")
    
    if save_embeddings:
        # Save as numpy arrays (fast loading for plotting)
        np.save(output_path / "train_embeddings.npy", train_embeddings)
        np.save(output_path / "train_labels.npy", train_labels)
        np.save(output_path / "test_embeddings.npy", test_embeddings)
        np.save(output_path / "test_labels.npy", test_labels)
        saved_files.extend(["train_embeddings.npy", "train_labels.npy",
                           "test_embeddings.npy", "test_labels.npy"])

        # Also save as Excel (compatible with AVG_Embd loading.py)
        train_excel = save_embeddings_to_excel(
            train_embeddings, train_sentences, train_labels, output_path, "train")
        test_excel = save_embeddings_to_excel(
            test_embeddings, test_sentences, test_labels, output_path, "test")
        saved_files.extend([train_excel.name, test_excel.name])
    
    # Save run config
    run_config = {
        'model_name': model_name,
        'pooling': pooling,
        'device': str(device),
        'normalize': normalize,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'early_stopping_patience': early_stopping_patience,
        'l2_lambda': l2_lambda,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'random_seed': random_seed,
        'results': {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'best_epoch': nn.history.get('best_epoch', epochs),
            'best_val_acc': nn.history.get('best_val_acc', float(test_acc)),
            'actual_epochs': len(nn.history['loss']),
        },
        'nn_separation_metrics': {
            'geometric_separability_index': float(nn_separation['geometric_separability_index']),
            'mann_whitney_p_value': float(nn_separation['mann_whitney_u']['p_value']),
            'mann_whitney_significant': nn_separation['mann_whitney_u']['significant'],
            'effect_size': float(nn_separation['mann_whitney_u']['effect_size']),
            'mean_class0': float(nn_separation['mean_score_class0']),
            'mean_class1': float(nn_separation['mean_score_class1']),
            'mean_difference': float(nn_separation['mean_difference'])
        },
        'semantic_axes': all_axes_results,
        'pairwise_evaluation': pairwise_results_all,
        'ridge_probe': ridge_results,
        'pca_analysis': pca_results
    }
    
    with open(output_path / "run_config.yaml", 'w') as f:
        yaml.dump(run_config, f, default_flow_style=False)
    saved_files.append("run_config.yaml")

    logger.info("results_saved", output_dir=str(output_path), files=saved_files)
    logger.info("training_complete")
    send_slack_notification(
        f"[Embd-NN] SL training complete — dataset={dataset_name}, "
        f"test_acc={test_acc:.4f}, output={output_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        send_slack_notification(f"[Embd-NN] SL training FAILED: {e}")
        raise
