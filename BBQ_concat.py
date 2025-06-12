import os
import json
import pandas as pd
from glob import glob

# Paths
bbq_data_path = "/cs/student/projects3/aisd/2024/ghanda/BBQ-main/data"
metadata_path = "/cs/student/projects3/aisd/2024/ghanda/BBQ-main/analysis_scripts/additional_metadata.csv"

# Load BBQ .jsonl files
def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

all_jsonls = glob(os.path.join(bbq_data_path, "*.jsonl"))

examples = []
for fpath in all_jsonls:
    examples.extend(load_jsonl(fpath))

df_bbq = pd.DataFrame(examples)

# Load and join metadata
df_meta = pd.read_csv(metadata_path)
df_full = pd.merge(df_bbq, df_meta, on="example_id", how="left")
print("Total BBQ examples:", len(df_bbq))
print("Merged rows:", len(df_full))
print("Missing metadata:", df_full['label_type'].isnull().sum())


# Filter ambiguous only
df_ambig = df_full[df_full["context_condition"] == "ambig"]

# Save to file
output_path = "/cs/student/projects3/aisd/2024/ghanda/bbq_ambiguous_with_metadata.csv"
df_ambig.to_csv(output_path, index=False)

print(f"Saved: {output_path}")

