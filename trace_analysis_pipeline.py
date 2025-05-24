# Trace Analysis Pipeline for Haruhi Prime Superpermutation Runs

import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
TRACE_DIR = "./traces"  # directory of trace_*.csv files

# --- Utility ---
def load_traces(trace_dir):
    trace_data = []
    for path in glob.glob(os.path.join(trace_dir, "trace_*.csv")):
        try:
            df = pd.read_csv(path)
            df['trace_file'] = os.path.basename(path)
            df['run_id'] = os.path.basename(path).split("trace_")[-1].replace(".csv", "")
            trace_data.append(df)
        except Exception as e:
            print(f"Failed to parse {path}: {e}")
    return trace_data

# --- Aggregator ---
def summarize_trace(df):
    def high_overlap_ratio(x):
        return np.mean(x >= 3)

    summary = {
        'run_id': df.run_id.iloc[0],
        'steps': len(df),
        'avg_overlap': df['overlap'].mean(),
        'max_overlap': df['overlap'].max(),
        'avg_entropy': df['entropy'].mean(),
        'entropy_std': df['entropy'].std(),
        'avg_shared_primes': df['shared_primes'].mean(),
        'avg_coherence_norm': df['coherence_norm'].mean(),
        'avg_step_time_ms': df['step_time_ms'].mean(),
        'avg_runtime_slope': df['runtime_slope'].mean(),
        'high_overlap_rate': high_overlap_ratio(df['overlap']),
        'final_entropy': df['entropy'].iloc[-1],
        'final_norm': df['coherence_norm'].iloc[-1],
        'final_overlap': df['overlap'].iloc[-1],
        'final_step': df['step'].iloc[-1]
    }
    return summary

# --- Correlation ---
def correlation_matrix(summary_df):
    numeric_df = summary_df.drop(columns=['run_id'])
    corr = numeric_df.corr(method='pearson')
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix of Run Summary Stats")
    plt.tight_layout()
    plt.show()

# --- Main Analysis ---
def analyze_all_traces():
    traces = load_traces(TRACE_DIR)
    summaries = [summarize_trace(df) for df in traces]
    summary_df = pd.DataFrame(summaries)
    print(summary_df.describe())

    # Save for future modeling
    summary_df.to_csv("trace_summaries.csv", index=False)

    # Correlation matrix
    correlation_matrix(summary_df)

    # Additional: Success classification, regression, etc. can build on summary_df

if __name__ == "__main__":
    analyze_all_traces()
