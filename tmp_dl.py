from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
repo = "google/timesfm-2.5-200m-pytorch"
path = hf_hub_download(repo_id=repo, filename="model.safetensors", revision="295319dbdc75b5205e139b284c4e14bcb56ced38")
print(path)
weights = load_file(path)
print(len(weights))
# inspect key name sample
for i,k in enumerate(weights.keys()):
    print(k)
    if i==20:
        break
