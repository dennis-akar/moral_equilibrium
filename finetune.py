"""
Step 5: Fine-tune GPT-4.1-mini on collected reflections via OpenAI API.
"""

import json
import time
import sys
from pathlib import Path

from openai import OpenAI


def upload_and_finetune(
    data_path: str = "data/finetune_data.jsonl",
    model: str = "gpt-4.1-mini-2025-04-14",
    suffix: str = "moral-equilibrium",
    n_epochs: int = 3,
):
    """Upload training file and start fine-tuning job."""
    client = OpenAI()

    # Validate data
    with open(data_path) as f:
        lines = f.readlines()
    print(f"Training file has {len(lines)} examples")

    if len(lines) < 10:
        print(f"WARNING: Only {len(lines)} training examples. OpenAI recommends at least 10.")
        print("Consider generating more reflections or reducing quality threshold.")

    # Upload file
    print("Uploading training file...")
    with open(data_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="fine-tune")
    print(f"File uploaded: {file_obj.id}")

    # Create fine-tuning job
    print(f"Starting fine-tuning job on {model}...")
    job = client.fine_tuning.jobs.create(
        training_file=file_obj.id,
        model=model,
        suffix=suffix,
        hyperparameters={"n_epochs": n_epochs},
    )
    print(f"Job created: {job.id}")
    print(f"Status: {job.status}")

    # Save job info
    job_info = {
        "job_id": job.id,
        "file_id": file_obj.id,
        "model": model,
        "status": job.status,
        "n_examples": len(lines),
        "n_epochs": n_epochs,
    }
    with open("data/finetune_job.json", "w") as f:
        json.dump(job_info, f, indent=2)

    return job


def check_status(job_id: str = None):
    """Check the status of a fine-tuning job."""
    client = OpenAI()

    if job_id is None:
        with open("data/finetune_job.json") as f:
            job_info = json.load(f)
        job_id = job_info["job_id"]

    job = client.fine_tuning.jobs.retrieve(job_id)
    print(f"Job: {job.id}")
    print(f"Status: {job.status}")
    if job.fine_tuned_model:
        print(f"Fine-tuned model: {job.fine_tuned_model}")
        # Save model name
        with open("data/finetune_job.json") as f:
            job_info = json.load(f)
        job_info["fine_tuned_model"] = job.fine_tuned_model
        job_info["status"] = job.status
        with open("data/finetune_job.json", "w") as f:
            json.dump(job_info, f, indent=2)
    if job.error:
        print(f"Error: {job.error}")

    # Show recent events
    events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=5)
    print("\nRecent events:")
    for event in events.data:
        print(f"  [{event.created_at}] {event.message}")

    return job


def wait_for_completion(job_id: str = None, poll_interval: int = 30):
    """Poll until fine-tuning job completes."""
    if job_id is None:
        with open("data/finetune_job.json") as f:
            job_info = json.load(f)
        job_id = job_info["job_id"]

    client = OpenAI()
    print(f"Waiting for job {job_id} to complete...")

    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"  Status: {job.status}")

        if job.status == "succeeded":
            print(f"\nFine-tuning complete!")
            print(f"Model: {job.fine_tuned_model}")
            # Save
            with open("data/finetune_job.json") as f:
                job_info = json.load(f)
            job_info["fine_tuned_model"] = job.fine_tuned_model
            job_info["status"] = "succeeded"
            with open("data/finetune_job.json", "w") as f:
                json.dump(job_info, f, indent=2)
            return job

        if job.status in ("failed", "cancelled"):
            print(f"\nJob {job.status}: {job.error}")
            return job

        time.sleep(poll_interval)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        job_id = sys.argv[2] if len(sys.argv) > 2 else None
        check_status(job_id)
    elif len(sys.argv) > 1 and sys.argv[1] == "wait":
        job_id = sys.argv[2] if len(sys.argv) > 2 else None
        wait_for_completion(job_id)
    else:
        upload_and_finetune()
