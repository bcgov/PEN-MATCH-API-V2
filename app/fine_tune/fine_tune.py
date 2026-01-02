import os
import time
import json
import requests
from pathlib import Path

from app.config import settings

# ---------- Config via settings (Key Vault) ----------
ENDPOINT = settings.openai_api_base_o4_mini_finetune
API_KEY = settings.openai_api_key
API_VERSION = "2025-04-16"

BASE_MODEL = "o4-mini"
TRAIN_FILE = Path(os.getenv("TRAIN_FILE", "fine_tune/train.jsonl"))
VAL_FILE = Path(os.getenv("VAL_FILE", "fine_tune/validation.jsonl"))
TEST_FILE = Path(os.getenv("TEST_FILE", "fine_tune/test.jsonl"))
SUFFIX = os.getenv("SUFFIX", "penmatch-lora")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "20"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "7200"))  # 2 hours


def _headers():
    return {
        "api-key": API_KEY,
        "Content-Type": "application/json"
    }


def upload_file(path: Path, purpose: str = "fine-tune") -> str:
    """
    Uploads a file to Azure OpenAI and returns file id.
    REST: POST {endpoint}/openai/files?api-version=2024-10-21
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    url = f"{ENDPOINT}/openai/files?api-version={API_VERSION}"

    # Azure OpenAI Files Upload uses multipart/form-data
    files = {
        "file": (path.name, path.read_bytes(), "application/jsonl"),
    }
    data = {
        "purpose": purpose,
    }

    # Remove Content-Type for multipart upload
    headers = {"api-key": API_KEY}
    resp = requests.post(url, headers=headers, data=data, files=files, timeout=120)
    
    if resp.status_code >= 400:
        raise RuntimeError(f"Upload failed ({resp.status_code}): {resp.text}")

    payload = resp.json()
    file_id = payload.get("id")
    if not file_id:
        raise RuntimeError(f"Upload response missing id: {payload}")
    
    print(f"✓ Uploaded {path.name} with file_id: {file_id}")
    return file_id


def create_finetune_job(training_file_id: str, validation_file_id: str | None = None, test_file_id: str | None = None) -> str:
    """
    Creates a fine-tuning job with LoRA method and returns the job id.
    REST: POST {endpoint}/openai/fine_tuning/jobs?api-version=2024-10-21
    """
    url = f"{ENDPOINT}/openai/fine_tuning/jobs?api-version={API_VERSION}"

    # Configure LoRA fine-tuning
    body = {
        "model": BASE_MODEL,
        "training_file": training_file_id,
        "suffix": SUFFIX,
        "method": {
            "type": "lora",  # Use LoRA instead of full fine-tuning
            "lora_config": {
                "rank": 16,  # LoRA rank - adjust as needed (8, 16, 32)
                "alpha": 32,  # LoRA alpha parameter
                "dropout": 0.1,  # LoRA dropout
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]  # Target attention layers
            }
        },
        "hyperparameters": {
            "n_epochs": 3,  # Number of training epochs
            "batch_size": 8,  # Batch size
            "learning_rate_multiplier": 1.0  # Learning rate multiplier
        }
    }
    
    if validation_file_id:
        body["validation_file"] = validation_file_id
    
    if test_file_id:
        body["test_file"] = test_file_id

    resp = requests.post(url, headers=_headers(), data=json.dumps(body), timeout=120)
    
    if resp.status_code >= 400:
        raise RuntimeError(f"Create job failed ({resp.status_code}): {resp.text}")

    payload = resp.json()
    job_id = payload.get("id")
    if not job_id:
        raise RuntimeError(f"Create job response missing id: {payload}")
    
    print(f"✓ Created fine-tuning job: {job_id}")
    return job_id


def get_job(job_id: str) -> dict:
    """
    Gets a fine-tune job status.
    REST: GET {endpoint}/openai/fine_tuning/jobs/{id}?api-version=...
    """
    url = f"{ENDPOINT}/openai/fine_tuning/jobs/{job_id}?api-version={API_VERSION}"
    resp = requests.get(url, headers={"api-key": API_KEY}, timeout=60)
    
    if resp.status_code >= 400:
        raise RuntimeError(f"Get job failed ({resp.status_code}): {resp.text}")
    
    return resp.json()


def validate_settings():
    """Validate that all required settings are available"""
    if not ENDPOINT:
        raise ValueError("openai_api_base_o4_mini_finetune not found in settings")
    if not API_KEY:
        raise ValueError("openai_api_key not found in settings")
    
    print(f"✓ Endpoint: {ENDPOINT}")
    print(f"✓ API Key: {'*' * (len(API_KEY) - 4) + API_KEY[-4:] if API_KEY else 'Missing'}")


def main():
    print("="*60)
    print("AZURE OPENAI GPT-4O-MINI LORA FINE-TUNING")
    print("="*60)
    
    # Validate configuration
    print("\n1. Validating configuration...")
    validate_settings()
    
    print(f"\nConfiguration:")
    print(f"- API Version: {API_VERSION}")
    print(f"- Base Model: {BASE_MODEL}")
    print(f"- Method: LoRA")
    print(f"- Suffix: {SUFFIX}")
    
    print(f"\nData files:")
    print(f"- Train file: {TRAIN_FILE} ({'✓' if TRAIN_FILE.exists() else '✗ Missing'})")
    print(f"- Validation file: {VAL_FILE} ({'✓' if VAL_FILE.exists() else '✗ Missing'})")
    print(f"- Test file: {TEST_FILE} ({'✓' if TEST_FILE.exists() else '✗ Missing'})")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file required: {TRAIN_FILE}")

    # Upload files
    print("\n2. Uploading files...")
    train_id = upload_file(TRAIN_FILE, purpose="fine-tune")

    val_id = None
    if VAL_FILE.exists():
        val_id = upload_file(VAL_FILE, purpose="fine-tune")
    else:
        print("⚠ No validation file found - proceeding without validation")

    test_id = None
    if TEST_FILE.exists():
        test_id = upload_file(TEST_FILE, purpose="fine-tune")
    else:
        print("⚠ No test file found - proceeding without test evaluation")

    # Create fine-tuning job
    print("\n3. Creating LoRA fine-tuning job...")
    job_id = create_finetune_job(train_id, val_id, test_id)

    # Poll for completion
    print(f"\n4. Monitoring job progress...")
    print(f"Job ID: {job_id}")
    print(f"Polling interval: {POLL_SECONDS} seconds")
    print(f"Timeout: {TIMEOUT_SECONDS} seconds")
    
    start = time.time()
    iteration = 0
    
    while True:
        iteration += 1
        job = get_job(job_id)
        status = job.get("status")
        elapsed = time.time() - start
        
        print(f"[{iteration:3d}] Status: {status:<12} | Elapsed: {elapsed:6.1f}s")
        
        # Show training progress if available
        if "training_progress" in job:
            progress = job["training_progress"]
            if isinstance(progress, dict):
                epoch = progress.get("current_epoch", 0)
                step = progress.get("current_step", 0)
                loss = progress.get("train_loss")
                print(f"      Progress: Epoch {epoch}, Step {step}" + 
                      (f", Loss: {loss:.4f}" if loss else ""))

        # Check for terminal statuses
        if status in {"succeeded", "failed", "cancelled"}:
            print(f"\n{'='*60}")
            print(f"FINE-TUNING {'SUCCESS' if status == 'succeeded' else 'FAILED'}")
            print(f"{'='*60}")
            
            if status == "succeeded":
                ft_model = job.get("fine_tuned_model")
                if ft_model:
                    print(f"✅ Fine-tuned model: {ft_model}")
                    print(f"✅ Model suffix: {SUFFIX}")
                    print(f"✅ Training method: LoRA")
                    print(f"\nNext steps:")
                    print(f"1. Deploy the model in Azure OpenAI Studio")
                    print(f"2. Update your application to use: {ft_model}")
                    print(f"3. Test the model with your validation data")
                else:
                    print("⚠ Succeeded but fine_tuned_model missing in response")
            else:
                error = job.get("error", "Unknown error")
                print(f"❌ Job failed: {error}")
                print(f"\nJob details:")
                print(json.dumps({k: job.get(k) for k in ["id", "status", "error", "created_at"]}, indent=2))
            
            break

        # Check timeout
        if elapsed > TIMEOUT_SECONDS:
            raise TimeoutError(f"Fine-tuning job timed out after {TIMEOUT_SECONDS}s (job: {job_id})")
        
        time.sleep(POLL_SECONDS)

    print(f"\nTotal time: {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise