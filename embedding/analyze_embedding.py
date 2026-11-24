import pandas as pd

# Read benchmark results
df = pd.read_csv("results/embedding_results.csv")

# Sort by model and run number
df = df.sort_values(["model", "run"])

# Columns to display
columns = [
    "model",
    "run",
    "latency_seconds",
    "ollama_total_seconds",
    "load_time_seconds",
    "texts_per_second",
    "embedding_dimensions"
]

# Store detailed results and averages
result_rows = []

# Process each model separately
for model, group in df.groupby("model", sort=False):

    # Add every individual run
    for _, row in group.iterrows():
        result_rows.append({
            "model": model,
            "run": f"Run {row['run']}",
            "latency_seconds": row["latency_seconds"],
            "ollama_total_seconds": row["ollama_total_seconds"],
            "load_time_seconds": row["load_time_seconds"],
            "texts_per_second": row["texts_per_second"],
            "embedding_dimensions": row["embedding_dimensions"]
        })

    # Calculate average
    result_rows.append({
        "model": model,
        "run": "AVG",
        "latency_seconds": group["latency_seconds"].mean(),
        "ollama_total_seconds": group["ollama_total_seconds"].mean(),
        "load_time_seconds": group["load_time_seconds"].mean(),
        "texts_per_second": group["texts_per_second"].mean(),
        "embedding_dimensions": group["embedding_dimensions"].iloc[0]
    })

# Create final report
summary = pd.DataFrame(result_rows)

# Round numeric results
numeric_columns = [
    "latency_seconds",
    "ollama_total_seconds",
    "load_time_seconds",
    "texts_per_second"
]

summary[numeric_columns] = summary[numeric_columns].round(3)

# Display results
print("\n========== EMBEDDING BENCHMARK SUMMARY ==========\n")
print(summary.to_string(index=False))

# Export CSV
summary.to_csv(
    "results/embedding_summary.csv",
    index=False
)

# Export Excel
summary.to_excel(
    "results/embedding_summary.xlsx",
    index=False
)

print("\nSummary exported successfully!")
