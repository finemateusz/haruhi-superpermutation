# Haruhi Prime-Coherence Superpermutation

This project implements a heuristic search algorithm to find short superpermutations, leveraging a unique "Prime-Coherence" scoring mechanism. A superpermutation for `n` symbols is a string that contains every possible permutation of those `n` symbols as a contiguous substring. Finding the *shortest* such string is a challenging combinatorial problem.

## Introduction

The Superpermutation Problem asks for the shortest string containing all `n!` permutations of `n` symbols. While solutions for small `n` are known, the problem becomes computationally intensive quickly. This project explores a novel heuristic approach guided by number-theoretic properties of permutations.

## Approach: The Prime-Coherence Heuristic

The core of this solver lies in its unique "coherence kernel" - a scoring function used to guide the selection of the next permutation in the sequence.

1.  **Lehmer Coding:** Permutations are first converted into unique integers using Lehmer codes.
2.  **UniversalNumber Abstraction:** Each Lehmer code is then treated as a `UniversalNumber`. This custom class calculates:
    *   **Prime Factorization:** The prime factors and their exponents.
    *   **Coherence Norm:** A weighted sum of prime factors (`sum(prime * exponent)`).
    *   **Factorial Entropy:** Shannon entropy of the distribution of prime factor exponents.
3.  **Coherence Kernel Scoring:** When deciding which permutation to add next, the `coherence_kernel` evaluates candidates based on a weighted combination of:
    *   **Overlap:** Maximizing the overlap with the previous permutation (crucial for shortness).
    *   **Shared Primes:** The number of common prime factors between the Lehmer codes of the current and candidate permutations.
    *   **Norm Alignment:** Similarity of their "coherence norms."
    *   **Entropy Stability:** Favoring candidates that maintain or smoothly evolve the running entropy of the path's Lehmer codes.
    *   **Probability Lookups:** Prioritizing permutations whose Lehmer code norm and entropy fall into regions deemed "more probable" based on predefined density functions (modelled as exponential decays).
4.  **Lookahead:** A 1-step lookahead mechanism is employed, meaning the algorithm considers not just the immediate best score but also the potential score of the step *after* the next, helping to avoid some local optima.

The intuition behind using prime-related metrics is to explore whether number-theoretic properties of permutation encodings can act as effective heuristics for guiding the search towards more structured and ultimately shorter superpermutations. This approach has successfully reproduced one of the known minimal superpermutations of length 153 for n=5. This result was originally found by Ben Chaffin in March 2014 and is documented by Nathaniel Johnston [here](https://www.njohnston.ca/superperm5.txt).

    *   123451234152341253412354123145231425314235142315423124531243512431524312543121345213425134215342135421324513241532413524132541321453214352143251432154321

This demonstrates the capability of the "Prime-Coherence" heuristic to guide the search towards optimal solutions for certain problem instances.

## Features

*   **Advanced Heuristic Search:** Employs the custom "Prime-Coherence" kernel.
*   **Multiprocessing:** Utilizes multiple CPU cores to speed up the evaluation of candidate permutations in parallel.
*   **Checkpointing & Resuming:**
    *   The main solver (`haruhi_coherence_kernel.py`) can save promising partial solutions (checkpoints) to `checkpoint_memory.json`.
    *   The resume script (`haruhi_resume_all_checkpoints.py`) can load these checkpoints and continue the search, allowing for deeper exploration of good paths or recovery from interruptions.
*   **Persistent Best Record:** Tracks the shortest superpermutation found for each `n` across all runs in `shortest_record.json`.
*   **Detailed Data Logging:**
    *   Comprehensive step-by-step metrics (overlap, entropy, norm, timing, etc.) are logged to CSV files in the `traces/` directory for each run.
    *   This allows for offline analysis and potential refinement of the heuristics.
*   **Trace Analysis Pipeline:** A dedicated script (`trace_analysis_pipeline.py`) to aggregate, summarize, and analyze the generated trace data (e.g., producing correlation matrices).

## File List & Usage

The project consists of three main Python scripts and several generated files/directories.

*   **`haruhi_coherence_kernel.py` (Main Solver)**
    *   **Purpose:** Finds superpermutations from scratch using multiple random seeds and the prime-coherence heuristic. Saves checkpoints and best results.
    *   **How to run:**
        ```bash
        python haruhi_coherence_kernel.py -n <number_of_symbols>
        ```
        Example: `python haruhi_coherence_kernel.py -n 4`

*   **`haruhi_resume_all_checkpoints.py` (Resumer)**
    *   **Purpose:** Loads saved checkpoints from `checkpoint_memory.json` and continues the search from those points.
    *   **How to run:**
        ```bash
        python haruhi_resume_all_checkpoints.py -n <number_of_symbols>
        ```
        Example: `python haruhi_resume_all_checkpoints.py -n 4` (ensure `checkpoint_memory.json` exists from a previous kernel run for the same `n`)

*   **`trace_analysis_pipeline.py` (Analyzer)**
    *   **Purpose:** Analyzes the `.csv` trace files generated by the solver and resumer.
    *   **How to run:**
        ```bash
        python trace_analysis_pipeline.py
        ```
        (This script typically doesn't require command-line arguments as it scans the `traces/` directory).

**Generated Files/Directories:**

*   **`traces/` (Directory):** Contains detailed CSV log files for each run/resume attempt (e.g., `trace_run_n4_...csv`).
*   **`checkpoint_memory.json` (File):** Stores checkpoint data from `haruhi_coherence_kernel.py` for promising partial solutions.
*   **`shortest_record.json` (File):** Keeps a record of the shortest superpermutation length found for each `n`, along with file paths to the solution and its trace.
*   **`attempts_summary.txt` (File):** A summary log of attempts made by `haruhi_coherence_kernel.py`.
*   **`trace_summaries.csv` (File):** Aggregated summary statistics from all trace files, generated by `trace_analysis_pipeline.py`.
*   **`best_path_n<N>_<length>_<timestamp>.txt` (File):** The shortest superpermutation string found.
*   **`best_trace_n<N>_<length>_<timestamp>.csv` (File):** The trace log corresponding to the best path.

### Example Optimal Result for n=5

The repository includes the following files demonstrating a successfully found minimal superpermutation for n=5:

*   **`best_path_n5_153_20250409_165450.txt`**: The actual superpermutation string of length 153.
*   **`best_trace_n5_153_20250409_165450.csv`**: The detailed step-by-step trace log from the run that found this optimal solution.

## Requirements

*   Python 3.x
*   The following Python libraries (can be installed via `pip install -r requirements.txt`):
    *   `numpy`
    *   `sympy`
    *   `pandas`
    *   `matplotlib`
    *   `seaborn`
