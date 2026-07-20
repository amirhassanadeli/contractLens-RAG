from datasets import load_dataset

print("Loading...")

from datasets import load_dataset

ds = load_dataset("theatticusproject/cuad-qa")

print(ds)
print(ds["train"][0])