from datasets import load_dataset

# 1. Authenticate (if dataset is gated)
from huggingface_hub import login
login()  # Run `huggingface-cli login` first if needed

# 2. Load dataset
ds = load_dataset("EPFL-ECEO/coralscapes", split="train")  # or "test"/"validation"

# 3. Access data (lazy loading)
print(ds[0])  # First sample
ds.save_to_disk("./coralscapes_data")  # Save locally