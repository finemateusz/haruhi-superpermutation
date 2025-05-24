# Prime-based Haruhi Superpermutation Solver

import datetime
import itertools
import math
import argparse
import time
from sympy import factorint
from multiprocessing import Pool, cpu_count, get_context
import numpy as np
from collections import Counter, deque
import random
import os
import json

trace_dir = "traces"
os.makedirs(trace_dir, exist_ok=True)

# === Prime Math Abstractions ===
class UniversalNumber:
    def __init__(self, value):
        self.value = value
        self.factors = factorint(value) if value > 0 else {1: 1}

    def shared_primes_with(self, other):
        return set(self.factors.keys()) & set(other.factors.keys())

    def coherence_norm(self):
        return sum(p * e for p, e in self.factors.items())

    def entropy(self):
        total = sum(self.factors.values())
        if total == 0:
            return 0
        probs = [v / total for v in self.factors.values()]
        return -sum(p * math.log(p) for p in probs if p > 0)

# === Chaffin Profile Histogram ===
def create_density_lookup(hist):
    def lookup(value):
        for i in range(len(hist["bins"]) - 1):
            if hist["bins"][i] <= value < hist["bins"][i + 1]:
                return max(hist["density"][i], 1e-6)
        return 1e-6
    return lookup

entropy_bins = np.linspace(0, 1.5, 20)
norm_bins = np.linspace(0, 100, 20)
entropy_hist = {"bins": entropy_bins, "density": np.exp(-entropy_bins[:-1])}
norm_hist = {"bins": norm_bins, "density": np.exp(-norm_bins[:-1] / 25)}
lookup_entropy = create_density_lookup(entropy_hist)
lookup_norm = create_density_lookup(norm_hist)

# === Known lower bounds (editable) ===
MINIMAL_LENGTHS = {
    5: 153,
    6: 872  # lowest FOUND, not proven minimal
}

# === Coherence Kernel ===
def coherence_kernel(p1, i1, f1, p2, running_entropy):
    i2 = lehmer_encode(p2)
    u1 = UniversalNumber(i1)
    u2 = UniversalNumber(i2)

    f2 = u2.factors
    shared = len(u1.shared_primes_with(u2))
    overlap = compute_overlap(p1, p2)
    norm_alignment = 1.0 if u1.coherence_norm() == u2.coherence_norm() else 0.5 if abs(u1.coherence_norm() - u2.coherence_norm()) < 50 else 0
    entropy2 = u2.entropy()
    entropy_drift = entropy2 - running_entropy

    log_p_entropy = math.log(lookup_entropy(entropy2))
    log_p_norm = math.log(lookup_norm(u2.coherence_norm()))

    score = (
        overlap * 10000 +
        shared * 1000 +
        norm_alignment * 500 -
        entropy_drift * 200 +
        log_p_entropy * 1000 +
        log_p_norm * 500
    )

    return score, overlap, i2, f2, entropy2, shared, u2.coherence_norm()

# === Utility ===
def lehmer_encode(perm):
    lehmer = []
    items = list(perm)
    ref = sorted(items)
    for i in range(len(perm)):
        idx = ref.index(items[i])
        lehmer.append(idx)
        ref.pop(idx)
    return sum(lehmer[i] * math.factorial(len(perm) - 1 - i) for i in range(len(perm)))

def compute_overlap(a, b):
    n = len(a)
    for i in range(1, n):
        if a[-i:] == b[:i]:
            return i
    return 0

def score_candidate(args):
    current, current_index, current_factors, cand, running_entropy = args
    return coherence_kernel(current, current_index, current_factors, cand, running_entropy) + (cand,)

# === Main Logic With Checkpoint Memory ===
def build_prime_superpermutation(perms, verbose=False, trace_file="trace-log.txt", max_retries=10):
    total = len(perms)
    # Dynamically scaled checkpoints (5%, 10%, 25%, 50%)
    checkpoint_steps = [int(total * p) for p in [0.05, 0.1, 0.25, 0.5]]
    best_result = None
    best_length = float('inf')
    attempt_log = []
    checkpoint_log = {}

    # === Load persistent best record if available ===
    record_file = "shortest_record.json"
    global_best = {}
    if os.path.exists(record_file):
        with open(record_file, "r") as rf:
            global_best = json.load(rf)
        key = f"n{len(perms[0])}"
        if key in global_best:
            best_length = global_best[key]["length"]

    target_length = MINIMAL_LENGTHS.get(len(perms[0]), None)

    for attempt in range(max_retries):
        seed_perm = random.choice(perms)
        current = seed_perm
        current_index = lehmer_encode(current)
        current_factors = factorint(current_index)
        visited = set([current])
        path = [current]
        sequence = list(current)
        entropy_history = []
        runtime_window = deque(maxlen=10)
        start_time = time.time()

        print(f"[Seed Attempt {attempt+1}] Starting from seed {seed_perm}\n")

        trace_lines = []
        trace_lines.append("step,index,overlap,shared_primes,entropy,coherence_norm,remaining_high_overlap,eta,step_time_ms,runtime_slope")
        trace_lines.append(f"Total permutations: {total}")
        trace_lines.append(f"Start permutation: {current} | Index: {current_index} | Factors: {current_factors}")

        with get_context("spawn").Pool(processes=cpu_count()) as pool:
                step = 0
                running_entropy = 0
                last_checkpoint = None

                while len(visited) < total:
                    step_start = time.time()
                    remaining = [p for p in perms if p not in visited]
                    args = [(current, current_index, current_factors, cand, running_entropy) for cand in remaining]
                    first_results = pool.map(score_candidate, args)

                    lookahead_scores = []
                    for first in first_results:
                        s1, o1, i1, f1, e1, sh1, n1, cand1 = first
                        rem2 = [p for p in remaining if p != cand1]
                        args2 = [(cand1, i1, f1, cand2, e1) for cand2 in rem2]
                        second_results = pool.map(score_candidate, args2)
                        best2 = max([r[0] for r in second_results], default=0)
                        total_score = s1 + 0.8 * best2
                        lookahead_scores.append((total_score, first))

                    best_total, best_first = max(lookahead_scores, key=lambda x: x[0])
                    _, best_overlap, best_index, best_factors, best_entropy, best_shared, best_norm, best_next = best_first

                    sequence.extend(best_next[best_overlap:])
                    visited.add(best_next)
                    path.append(best_next)

                    entropy_history.append(best_entropy)
                    running_entropy = sum(entropy_history[-10:]) / min(len(entropy_history), 10)

                    step_time_ms = (time.time() - step_start) * 1000
                    runtime_window.append(step_time_ms)
                    runtime_slope = 0 if len(runtime_window) < 2 else (runtime_window[-1] - runtime_window[0]) / len(runtime_window)

                    high_overlap_count = sum(1 for r in first_results if r[1] >= 3)
                    progress = len(visited) / total
                    elapsed = time.time() - start_time
                    eta = (elapsed / progress * (1 - progress)) if progress > 0 else 0

                    log = f"→ {best_next} | Overlap: {best_overlap} | ETA: {eta:.1f}s | Index: {best_index} | Factors: {best_factors} | Step Time: {step_time_ms:.1f}ms\n"
                    trace_lines.append(f"{step},{best_index},{best_overlap},{best_shared},{best_entropy:.4f},{best_norm},{high_overlap_count},{eta:.2f},{step_time_ms:.1f},{runtime_slope:.4f}")
                    trace_lines.append(log)
                    if verbose:
                        print(log.strip())

                    current = best_next
                    current_index = best_index
                    current_factors = best_factors
                    step += 1

                    if step in checkpoint_steps:
                        if best_entropy > 0.9 or runtime_slope >= 1.0:
                            print(f"[Checkpoint @ step {step}] ✘ Coherence not forming (entropy: {best_entropy:.2f}, slope: {runtime_slope:.2f}) — restarting path...\n")
                            break
                        else:
                            last_checkpoint = {
                                "seed": seed_perm,
                                "step": step,
                                "entropy": best_entropy,
                                "runtime_slope": runtime_slope,
                                "path_so_far": path[:],
                                "sequence_so_far": sequence[:],
                                "visited_count": len(visited)
                            }
                            print(f"[Checkpoint @ step {step}] ✔ Coherence forming (entropy: {best_entropy:.2f}, slope: {runtime_slope:.2f})\n")

                result_len = len(sequence)

                # Always save full trace log to /traces/
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                nkey = f"n{len(perms[0])}"
                trace_all = os.path.join(trace_dir, f"trace_run_{nkey}_{result_len}_{timestamp}.csv")
                with open(trace_all, "w", encoding="utf-8") as tf:
                    tf.write("step,index,overlap,shared_primes,entropy,coherence_norm,remaining_high_overlap,eta,step_time_ms,runtime_slope\n")
                    tf.write("\n".join(trace_lines))

                attempt_log.append(f"Attempt {attempt+1}: length {result_len}, success: {len(visited)==total}")

                if last_checkpoint:
                    checkpoint_log[f"attempt_{attempt+1}"] = last_checkpoint

                if len(visited) == total:
                    print(f"[✔ Attempt {attempt+1}] Path complete (length: {result_len})\n")
                    if result_len < best_length:
                        best_result = (sequence, path)
                        best_length = result_len

                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        nkey = f"n{len(perms[0])}"

                        # Save best path
                        file_name = f"best_path_{nkey}_{result_len}_{timestamp}.txt"
                        with open(file_name, "w") as out:
                            out.write("".join(str(x) for x in sequence))

                        # Save best trace log to root (for quick access)
                        trace_name = f"best_trace_{nkey}_{result_len}_{timestamp}.csv"
                        with open(trace_name, "w", encoding="utf-8") as tf:
                            tf.write("step,index,overlap,shared_primes,entropy,coherence_norm,remaining_high_overlap,eta,step_time_ms,runtime_slope\n")
                            tf.write("\n".join(trace_lines))
                        
                        # Also save all trace logs to /traces/
                        trace_all = os.path.join(trace_dir, f"trace_run_{nkey}_{result_len}_{timestamp}.csv")
                        with open(trace_all, "w", encoding="utf-8") as tf:
                            tf.write("step,index,overlap,shared_primes,entropy,coherence_norm,remaining_high_overlap,eta,step_time_ms,runtime_slope\n")
                            tf.write("\n".join(trace_lines))

                        # Update global best record
                        global_best[nkey] = {
                            "length": result_len,
                            "timestamp": timestamp,
                            "path_file": file_name,
                            "trace_file": trace_name
                        }
                        with open(record_file, "w") as rf:
                            json.dump(global_best, rf, indent=2)

                    if target_length and result_len <= target_length:
                        print(f"✅ Target length {target_length} reached — stopping early")
                        break
                else:
                    print("[Restarting run from new seed permutation...]\n")


    with open("attempts_summary.txt", "w") as logf:
        logf.write("\n".join(attempt_log))

    with open("checkpoint_memory.json", "w") as ckpt:
        json.dump(checkpoint_log, ckpt, indent=2)

    if best_result:
        return best_result
    else:
        print("\n=== FINAL SUMMARY ===")
        print(f"Total successful paths found: {len([l for l in attempt_log if 'success: True' in l])}")
        sorted_logs = sorted(
            [l for l in attempt_log if 'success: True' in l],
            key=lambda x: int(x.split('length ')[1].split(',')[0])
        )
        for log in sorted_logs:
            print("•", log)
        print("\nNo shorter path found than current record. Existing shortest length remains unchanged.")

# === Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Haruhi Superpermutation Solver with Checkpoint Memory")
    parser.add_argument("-n", type=int, help="Number of symbols (e.g., 5 for n=5)")
    args = parser.parse_args()
    n = args.n

    symbols = list(range(1, n + 1))
    all_perms = list(itertools.permutations(symbols))
    print(f"Total permutations: {len(all_perms)}")

    start = time.time()

    try:
        result = build_prime_superpermutation(
            all_perms,
            verbose=True,
            trace_file=f"trace_kernel_n{n}.csv"
        )
        end = time.time()

        if result:
            superperm, perm_path = result
            print("\n=== FINAL SUMMARY ===")
            print(f"Total Permutations Used: {len(perm_path)}")
            print(f"Sequence Length: {len(superperm)}")
            print(f"Runtime: {end - start:.2f} seconds")
            print("\nFinal Path (last 10):")
            for p in perm_path[-10:]:
                print(p, "→", lehmer_encode(p), "→", factorint(lehmer_encode(p)))


    except RuntimeError as e:
        print(f"\n❌ {str(e)}")
        print("Tip: Increase `max_retries`, lower coherence thresholds, or use checkpoint resumes to explore saved paths.")

