import yaml
from src import logger


def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("config_loaded", path=config_path)
        return config
    except FileNotFoundError:
        logger.error("config_not_found", path=config_path)
        return None
    except Exception as e:
        logger.error("config_load_error", path=config_path, error=str(e))
        return None


def get_config_value(config, *keys, default=None):
    """Safely get nested config value"""
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value if value is not None else default
