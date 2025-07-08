import argparse, torch, yaml, datetime
from src.utils import save_results, clean_output_files, logger

from src.utils import send_slack_notification
from src.evaluation import loading, analysing
from src.embedding import embed_sentences, embed_group_1, embed_group_2
from src.methods import (
    extract_cls_pooling_st, extract_max_pooling_st,
    extract_mean_pooling_st, extract_mean_sqrt_len_pooling_st,
    extract_weightedmean_pooling_st, extract_lasttoken_pooling_st
    )

torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Process text with a specified model.')
parser.add_argument('--config', type=str, help='Path to the configuration YAML file')
parser.add_argument('--method', type=str, help='The name of the method to use')
parser.add_argument('--model_name', type=str, help='The name or path of the model to use')
parser.add_argument('--mode', type=str, default="danger", help='The mode to use (default: danger)')
parser.add_argument('--input_file', type=str, help='The name of the file to import')
args = parser.parse_args()


def process_configuration(model_ckpt, method, script):
    METHOD = method
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #--------------------START-----------------------  
    if script == 'danger':
        running_for = 'danger'
        from src.data.dict.danger import sentence_pairs         
    
    elif script == 'size':
        running_for = 'size'
        from src.data.dict.size import sentence_pairs    
    else:
        raise ValueError(f"Unknown script: {script}")
    #---------------------END-----------------------
    
    #--------------------START----------------------- 
    def process_batch_extra(batch):
        
        if METHOD == 'cls_pooling_st':
            return extract_cls_pooling_st(batch, model_ckpt, device)
        if METHOD == 'max_pooling_st':
            return extract_max_pooling_st(batch, model_ckpt, device)
        if METHOD == 'mean_pooling_st':
            return extract_mean_pooling_st(batch, model_ckpt, device)
        if METHOD == 'mean_sqrt_len_pooling_st':
            return extract_mean_sqrt_len_pooling_st(batch, model_ckpt, device)
        if METHOD == 'weightedmean_pooling_st':
            return extract_weightedmean_pooling_st(batch, model_ckpt, device)
        if METHOD == 'lasttoken_pooling_st':
            return extract_lasttoken_pooling_st(batch, model_ckpt, device)
        else:
            raise ValueError(f"Unknown method: {METHOD}")
    #---------------------END-----------------------
    
    #--------------------START-----------------------   
    logger.info("\n=== Initialization ===")
    logger.warning("Using model:", model=model_ckpt)
    logger.warning("Using input file:", running_for=running_for, with_method=METHOD)
    logger.warning("Using MODE:", mode = args.mode)
    logger.info("✓ Arguments processed successfully")
    logger.info("Using device:", device=device)
    #---------------------END----------------------- 

    #--------------------START----------------------- 
    if METHOD in ['cls_pooling_st', 'max_pooling_st',
                  'mean_pooling_st', 'mean_sqrt_len_pooling_st',
                  'weightedmean_pooling_st', 'lasttoken_pooling_st']:
        
        logger.warning("Using SentenceTransformer for tokenization and embedding extraction")
        dat_sentence = sentence_pairs
        
        if args.mode == "danger":
            logger.warning("You are using the DANGER mode")
            dat_group1 = ['safe', 'harmless', 'calm']
            dat_group2 = ['dangerous', 'deadly','threatening']
        
        elif args.mode == "size":
            logger.warning("You are using the SIZE mode")
            dat_group1 = ['small', 'little','tiny']
            dat_group2 = ['large', 'big', 'huge']
        
        else:
            raise ValueError(f"This mode isn't set up yet: {args.mode}") 
        
        # ========== Sentence Embeddings, Group 1 and Group 2 Embeddings ========== 
        dat_sentence = [process_batch_extra([item]) for item in dat_sentence]
        dat_group1 = [process_batch_extra([item]) for item in dat_group1]
        dat_group2 = [process_batch_extra([item]) for item in dat_group2]
       #---------------------END----------------------- 

    else:
        raise ValueError(f"Unknown method: {METHOD}")

    #--------------------START-----------------------
    # Create a folder name based on model, method, and dataset
    output_prefix = f"{model_ckpt.replace('/', '_')}_{METHOD}_{running_for}"
    file_path_sentences = embed_sentences(dat_sentence, METHOD, output_prefix)
    file_path_gr1 = embed_group_1(dat_group1, output_prefix)
    file_path_gr2 = embed_group_2(dat_group2, output_prefix)
    #---------------------END----------------------- 

    logger.info("\n=== Processing Complete ===")
    logger.info("✓ All embeddings have been successfully extracted and saved")

    #--------------------START-----------------------
    sentences_v1_embeddings, safe_group1_embeddings, danger_group2_embeddings = loading(
        file_path_sentences, file_path_gr1, file_path_gr2)

    result = analysing(
        sentences_v1_embeddings,
        safe_group1_embeddings,
        danger_group2_embeddings,
        running_for,
        METHOD
    )
    #---------------------END----------------------- 
    
    return result



if __name__ == "__main__":
    if args.config:
        with open(args.config, 'r') as config_file:
            config = yaml.safe_load(config_file)
        
        models = config.get('models', [])
        methods = config.get('methods', [])
        datasets = config.get('datasets', [])
        
        if 'mode' in config and config['mode']:
            args.mode = config['mode'][0] if isinstance(config['mode'], list) else config['mode']
        
        clean_output_files()
        
        results = {}
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for model in models:
            model_results = {}
            for method in methods:
                method_results = {}
                for dataset in datasets:
                    logger.info("\n\n=========================================================")
                    logger.info(f"Processing: Model={model}, Method={method}, Dataset={dataset}")
                    send_slack_notification(
                        f"Processing: Model={model}, Method={method}, Dataset={dataset}"
                    )
                    logger.info("=========================================================\n")
                    
                    try:
                        result = process_configuration(model, method, dataset)
                        method_results[dataset] = {
                            "metrics": result,
                            "timestamp": timestamp
                        }
                        logger.info(f"Successfully processed {model}/{method}/{dataset}")
                        send_slack_notification(f"Successfully processed {model}/{method}/{dataset}")
                    
                    except Exception as e:
                        send_slack_notification(
                            f"Error processing {model}/{method}/{dataset}: {e}"
                        )
                        logger.error(f"Error processing {model}/{method}/{dataset}: {e}")
                        method_results[dataset] = {
                            "error": str(e),
                            "timestamp": timestamp
                        }
                    clean_output_files()
                
                if method_results:
                    model_results[method] = method_results
            
            if model_results:
                model_key = model.replace('/', '_')
                results[model_key] = model_results
        
        save_results(results)
        logger.info("\n=== All Configurations Processed ===")
        
    else:
        # Use command line arguments for a single run
        if not all([args.method, args.model_name, args.input_file]):
            parser.error("When not using a config file, --method, --model_name, and --input_file are required")
        
        # Process the single configuration
        result = process_configuration(args.model_name, args.method, args.input_file)
        
        # Save result to output.yaml
        model_key = args.model_name.replace('/', '_')
        results = {
            model_key: {
                args.method: {
                    args.input_file: {
                        "metrics": result,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }}}}
        save_results(results)