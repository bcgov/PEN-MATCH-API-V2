import requests
import json
from config.settings import settings

ENDPOINT = "https://pen-match-api-v2-openai-canadaeast.openai.azure.com".rstrip("/")
API_KEY = settings.openai_api_key
API_VERSION = "2024-10-21"

def check_available_models():
    """Check what models are available for fine-tuning"""
    url = f"{ENDPOINT}/openai/models?api-version={API_VERSION}"
    headers = {"api-key": API_KEY}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return
    
    models = response.json()
    
    print("Available models:")
    print("="*50)
    
    finetune_models = []
    for model in models.get('data', []):
        model_id = model.get('id', '')
        capabilities = model.get('capabilities', {})
        
        # Check if fine-tuning is supported
        if capabilities.get('fine_tune', False):
            finetune_models.append(model_id)
            print(f"✓ {model_id} (supports fine-tuning)")
        elif 'gpt-4o-mini' in model_id.lower() or 'gpt-4' in model_id.lower():
            print(f"✗ {model_id} (no fine-tuning support)")
    
    print(f"\nModels that support fine-tuning:")
    for model in finetune_models:
        print(f"  - {model}")
    
    if not finetune_models:
        print("No models found that support fine-tuning!")
        print("\nFull response:")
        print(json.dumps(models, indent=2))

if __name__ == "__main__":
    check_available_models()