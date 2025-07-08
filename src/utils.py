import os, yaml, structlog, time
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

__all__ = [
    "save_results", 
    "clean_output_files", 
    "logger", 
    "HF_TOKEN",
    "send_slack_notification",
    ]



# Load credentials from credentials.yaml
with open(os.path.join(os.path.dirname(__file__), '..', 'credentials.yaml'), 'r') as cred_file:
    creds = yaml.safe_load(cred_file)

logger = structlog.get_logger()
HF_TOKEN = creds.get("HF_TOKEN")
OUTPUT_YAML = creds.get("OUTPUT_YAML", "output.yaml")
SLACK_BOT_TOKEN = creds.get("SLACK_BOT_TOKEN")
SLACK_CHANNEL = creds.get("SLACK_CHANNEL")
slack_client = WebClient(token=SLACK_BOT_TOKEN)


def send_slack_notification(message):
    """Send a notification to Slack channel via Slack Web API with retry logic."""
    for attempt in range(3):
        try:
            slack_client.chat_postMessage(channel=SLACK_CHANNEL, text=message)
            return
        except SlackApiError as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Failed to send Slack notification after 3 attempts: {e.response['error']}")
                raise e


def load_existing_results():
    """Load existing results from output.yaml if it exists"""
    if os.path.exists(OUTPUT_YAML):
        with open(OUTPUT_YAML, 'r') as f:
            try:
                return yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"Error loading existing results: {e}")
                return {}
    return {}

def save_results(results):
    """Save results to output.yaml"""
    existing_results = load_existing_results()
    merged_results = {**existing_results, **results}
    with open(OUTPUT_YAML, 'w') as f:
        yaml.dump(merged_results, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Results saved to {OUTPUT_YAML}")

def clean_output_files():
    output_dir = './src/data/'
    for file in os.listdir(output_dir):
        if file.endswith('.xlsx'):
            file_path = os.path.join(output_dir, file)
            try:
                os.remove(file_path)
                logger.info("Removed previous output file:", file_path=file_path)
            except Exception as e:
                logger.error(f"Error removing file {file_path}: {e}")
