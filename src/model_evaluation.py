import json

with open("models/model_metrics.json", "r") as f:
    metrics = json.load(f)

print("Final Model Evaluation Metrics")
for key, value in metrics.items():
    print(f"{key}: {value}")