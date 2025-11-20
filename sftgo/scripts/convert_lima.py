#!/usr/bin/env python
import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_messages(raw_messages):
    """
    Ensure messages is a plain Python list of {role, content} dicts.
    Your parquet stores it as a numpy array of dicts.
    """
    if isinstance(raw_messages, np.ndarray):
        raw_messages = raw_messages.tolist()

    if not isinstance(raw_messages, (list, tuple)):
        return None

    messages = []
    for m in raw_messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if not role or not content:
            continue
        # Keep only roles your encoder knows about
        if role not in ("system", "user", "assistant"):
            # you can skip or map other roles here if needed
            continue
        messages.append({"role": role, "content": content})

    return messages if messages else None


def convert_parquet_to_jsonl(parquet_path: str, jsonl_path: str):
    parquet_path = Path(parquet_path)
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if "messages" not in df.columns:
        raise ValueError(f"'messages' column not found in {parquet_path}")

    n_total = 0
    n_written = 0

    with jsonl_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            n_total += 1
            raw_messages = row["messages"]
            messages = normalize_messages(raw_messages)
            if not messages:
                continue  # skip rows with no usable messages

            record = {"messages": messages}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Done. Read {n_total} rows, wrote {n_written} JSONL lines to {jsonl_path}")


if __name__ == "__main__":
    # adjust these paths to your actual files
    parquet_path = "/mnt/data/lima.parquet"
    jsonl_path = "/mnt/data/lima.jsonl"

    convert_parquet_to_jsonl(parquet_path, jsonl_path)
