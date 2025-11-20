import pandas as pd
import json

# read sorted.csv produced by that script
scores = pd.read_csv("../output/sorted.csv")  # has header: file name, index, score
scores.columns = ["file_name", "index", "score"]

def load_jsonl_with_index(path, file_name):
    rows = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            rows.append({
                "file_name": file_name,
                "index": i,                 # 0-based, matches script
                "messages": obj["messages"]
            })
    return pd.DataFrame(rows)

# example for lima only
messages_df = load_jsonl_with_index("/mnt/data/lima.jsonl", "lima")

merged = messages_df.merge(scores, on=["file_name", "index"], how="inner")

merged.to_parquet("lima_less_selected_with_scores.parquet", index=False)
