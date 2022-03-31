import json

import requests

model_name = "facebook/rag-token-nq"
API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


result = query("When was Franklin D. Roosevelt born?")

print(result)
