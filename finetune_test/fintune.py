from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import time
client = OpenAI()

file = client.files.create(
    file=open("train.jsonl", "rb"),
    purpose="fine-tune"
)

job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-3.5-turbo"
)

job_id = job.id

while True:
    job = client.fine_tuning.jobs.retrieve(job_id)
    print("Status:", job.status)

    if job.status == "failed":
        print("❌ Error:", job.error)
        break

    if job.status in ["succeeded", "failed"]:
        print("Done:", job.fine_tuned_model)
        break

    time.sleep(10)