import pandas as pd
import polars as pl
import json

# Read sorted.csv produced by the scoring script
scores = pd.read_csv("../output/sorted.csv")  # has header: file name, index, score
scores.columns = ["file_name", "index", "score"]

# Filter for only lima entries
lima_scores = scores[scores["file_name"] == "lima"].copy()
print(f"Found {len(lima_scores)} lima entries in scores")

# Read the full lima parquet with all columns
lima_full = pd.read_parquet("/mnt/lima_chunked_128k_llama_3.2_token_importances_under_10_tokens_adjusted.parquet")
print(f"Loaded {len(lima_full)} rows from full lima parquet")
print(f"Columns: {lima_full.columns.tolist()}")

# Add an index column to match with scores (0-based)
lima_full["index"] = range(len(lima_full))
lima_full["file_name"] = "lima"

# Merge to get only the selected rows with scores, preserving all original columns
merged = lima_full.merge(lima_scores, on=["file_name", "index"], how="inner")

print(f"Selected {len(merged)} rows after merging")
print(f"Final columns: {merged.columns.tolist()}")

# Verify all original columns are preserved
expected_cols = [
    "messages", "source", "answers", 
    "llama_3.2_3b_chunks", "llama_3.2_3b_token_lengths",
    "llmlingua_token_lengths", "aggregated_tokens", "aggregated_probs",
    "file_name", "index", "score"
]

for col in expected_cols:
    if col not in merged.columns:
        print(f"WARNING: Missing column: {col}")

# Save to parquet
output_path = "lima_less_selected_with_scores_probs.parquet"
merged.to_parquet(output_path, index=False)
print(f"Saved to {output_path}")

# Optional: Print some stats
print(f"\nScore statistics:")
print(merged["score"].describe())
print(f"\nSample of first few rows:")
print(merged.head())

# Create top percentile subsets
print("\n" + "="*60)
print("Creating top percentile subsets...")
print("="*60)

percentiles = [5, 10, 20, 30, 40, 50]

for pct in percentiles:
    # Calculate the quantile threshold (e.g., top 10% means >= 0.90 quantile)
    quantile_threshold = (100 - pct) / 100
    threshold_score = merged["score"].quantile(quantile_threshold)
    
    # Filter and sort
    top_pct_df = merged[merged["score"] >= threshold_score].sort_values("score", ascending=False)
    
    # Save to parquet
    output_file = f"lima_less_selected_top{pct}.parquet"
    top_pct_df.to_parquet(output_file, index=False)
    
    print(f"\nTop {pct}%:")
    print(f"  - Threshold score: {threshold_score:.6f}")
    print(f"  - Number of rows: {len(top_pct_df)}")
    print(f"  - Score range: [{top_pct_df['score'].min():.6f}, {top_pct_df['score'].max():.6f}]")
    print(f"  - Saved to: {output_file}")

print("\n" + "="*60)
print("All files created successfully!")
print("="*60)