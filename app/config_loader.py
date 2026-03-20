import json
import os
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path='config.json'):
    """
    Load configuration from a JSON file.
    This version removes all Ollama/LLM-related logic since the pipeline is now YOLO-only.
    """
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        logger.info(f"Loaded configuration from {config_path}")

        # Expected config sections now:
        # db_config
        # s3_config
        # visicooler_config
        # activation_config

        return config

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def load_yaml_classes(yaml_path):
    """Load class IDs from a YAML file."""
    try:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)

        class_ids = yaml_data.get('names', {})

        if not class_ids:
            raise ValueError(f"No 'names' field found in YAML file: {yaml_path}")

        if isinstance(class_ids, list):
            class_ids = {str(i): name for i, name in enumerate(class_ids)}

        logger.info(f"Loaded class IDs from {yaml_path}")
        return class_ids

    except Exception as e:
        logger.error(f"Failed to load class IDs from {yaml_path}: {e}")
        raise


def load_json_classes(json_path):
    """Load class IDs from a JSON file."""
    try:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            class_ids = json.load(f)

        if not isinstance(class_ids, dict):
            raise ValueError(f"Invalid JSON format in {json_path}: Expected a dictionary")

        logger.info(f"Loaded class IDs from {json_path}")
        return class_ids

    except Exception as e:
        logger.error(f"Failed to load class IDs from {json_path}: {e}")
        raise