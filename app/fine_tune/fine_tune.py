import os
import time
import json
import requests
from pathlib import Path

from config.settings import settings

# ---------- Config via settings (Key Vault) ----------
ENDPOINT = "https://pen-match-api-v2-openai-canadaeast.openai.azure.com".rstrip("/")
API_KEY = settings.openai_api_key

# Use the stable data-plane API version for Files + Fine-tuning Jobs
API_VERSION = "2024-10-21"

# Use the fully-qualified base model name
BASE_MODEL = "gpt-4o-mini-2024-07-18"

TRAIN_FILE = Path(os.getenv("TRAIN_FILE", "fine_tune/train.jsonl"))
VAL_FILE = Path(os.getenv("VAL_FILE", "fine_tune/validation.jsonl"))
TEST_FILE = Path(os.getenv("TEST_FILE", "fine_tune/test.jsonl"))  # for your own offline eval only
SUFFIX = os.getenv("SUFFIX", "penmatch-lora")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "20"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "7200"))  # 2 hours


def _json_headers():
    return {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }


def _key_headers():
    # For GET or multipart upload (requests will set multipart content-type)
    return {"api-key": API_KEY}


def validate_settings():
    if not ENDPOINT:
        raise ValueError("AZURE OPENAI endpoint is empty")
    if "/openai/deployments/" in ENDPOINT:
        raise ValueError("ENDPOINT must be the resource root (no /openai/deployments/... path).")
    if not API_KEY:
        raise ValueError("openai_api_key not found in settings")

    print(f"✓ Endpoint: {ENDPOINT}")
    masked = ("*" * max(0, len(API_KEY) - 4)) + API_KEY[-4:]
    print(f"✓ API Key: {masked}")


def upload_file(path: Path, purpose: str = "fine-tune") -> str:
    """
    Uploads a file to Azure OpenAI and returns file id.
    REST: POST {endpoint}/openai/files?api-version=2024-10-21
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    url = f"{ENDPOINT}/openai/files?api-version={API_VERSION}"

    # multipart/form-data fields:
    # - purpose
    # - file
    files = {
        "file": (path.name, path.read_bytes(), "application/jsonl"),
    }
    data = {"purpose": purpose}

    resp = requests.post(url, headers=_key_headers(), data=data, files=files, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"Upload failed ({resp.status_code}): {resp.text}")

    payload = resp.json()
    file_id = payload.get("id")
    if not file_id:
        raise RuntimeError(f"Upload response missing id: {payload}")

    print(f"✓ Uploaded {path.name} with file_id: {file_id}")
    return file_id


def create_finetune_job(training_file_id: str, validation_file_id: str | None = None) -> str:
    """
    Creates a fine-tuning job and returns the job id.
    REST: POST {endpoint}/openai/fine_tuning/jobs?api-version=2024-10-21
    Body fields (per spec): model, training_file, validation_file?, suffix? 
    """
    url = f"{ENDPOINT}/openai/fine_tuning/jobs?api-version={API_VERSION}"

    body = {
        "model": BASE_MODEL,
        "training_file": training_file_id,
        "suffix": SUFFIX,
    }
    if validation_file_id:
        body["validation_file"] = validation_file_id

    resp = requests.post(url, headers=_json_headers(), data=json.dumps(body), timeout=120)
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
    REST: GET {endpoint}/openai/fine_tuning/jobs/{id}?api-version=2024-10-21 
    """
    url = f"{ENDPOINT}/openai/fine_tuning/jobs/{job_id}?api-version={API_VERSION}"
    resp = requests.get(url, headers=_key_headers(), timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"Get job failed ({resp.status_code}): {resp.text}")
    return resp.json()


def main():
    print("=" * 60)
    print("AZURE OPENAI GPT-4O-MINI FINE-TUNING (MANAGED LORA)")
    print("=" * 60)

    print("\n1) Validating configuration...")
    validate_settings()

    print("\nConfiguration:")
    print(f"- API Version: {API_VERSION}")
    print(f"- Base Model: {BASE_MODEL}")
    print(f"- Suffix: {SUFFIX}")

    print("\nData files:")
    print(f"- Train file: {TRAIN_FILE} ({'✓' if TRAIN_FILE.exists() else '✗ Missing'})")
    print(f"- Validation file: {VAL_FILE} ({'✓' if VAL_FILE.exists() else '✗ Missing'})")
    print(f"- Test file (offline eval only): {TEST_FILE} ({'✓' if TEST_FILE.exists() else '—'})")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file required: {TRAIN_FILE}")

    print("\n2) Uploading files...")
    train_id = upload_file(TRAIN_FILE, purpose="fine-tune")

    val_id = None
    if VAL_FILE.exists():
        val_id = upload_file(VAL_FILE, purpose="fine-tune")
    else:
        print("⚠ No validation file found — proceeding without validation")

    print("\n3) Creating fine-tuning job...")
    job_id = create_finetune_job(train_id, val_id)

    print("\n4) Monitoring job progress...")
    print(f"Job ID: {job_id}")
    print(f"Polling interval: {POLL_SECONDS}s | Timeout: {TIMEOUT_SECONDS}s")

    start = time.time()
    iteration = 0

    while True:
        iteration += 1
        job = get_job(job_id)
        status = job.get("status", "unknown")
        elapsed = time.time() - start

        ft_model = job.get("fine_tuned_model")
        print(f"[{iteration:3d}] Status: {status:<12} | Elapsed: {elapsed:7.1f}s"
              + (f" | fine_tuned_model: {ft_model}" if ft_model else ""))

        if status in {"succeeded", "failed", "cancelled"}:
            print("\n" + "=" * 60)
            print(f"FINE-TUNING RESULT: {status.upper()}")
            print("=" * 60)

            if status == "succeeded":
                if ft_model:
                    print(f"Fine-tuned model name: {ft_model}")
                    print("\nNext steps:")
                    print("1) Deploy the fine-tuned model (portal or deployment API).")
                    print("2) Update your app to call the deployment name for this fine-tuned model.")
                    print("3) Run your offline eval using test.jsonl to compare before/after.")
                else:
                    print("⚠ Succeeded but 'fine_tuned_model' missing. Print full job:")
                    print(json.dumps(job, indent=2))
            else:
                print("Job did not succeed. Error (if any):")
                print(json.dumps(job.get("error", {"detail": "No error field"}), indent=2))
                print("\nFull job (trimmed keys):")
                print(json.dumps({k: job.get(k) for k in ["id", "status", "error", "created_at"]}, indent=2))
            break

        if elapsed > TIMEOUT_SECONDS:
            raise TimeoutError(f"Fine-tuning job timed out after {TIMEOUT_SECONDS}s (job: {job_id})")

        time.sleep(POLL_SECONDS)

    print(f"\nTotal monitoring time: {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        raise
