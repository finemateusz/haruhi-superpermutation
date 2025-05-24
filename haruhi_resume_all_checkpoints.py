# Haruhi Prime-Coherence Superpermutation Resumer

import itertools
import math
import time
import json
import os
import datetime
import numpy as np
from sympy import factorint
from collections import deque
from multiprocessing import Pool, cpu_count, get_context

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

# === Utilities ===
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

# === Coherence Kernel Setup ===
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

def score_candidate(args):
    current, current_index, current_factors, cand, running_entropy = args
    return coherence_kernel(current, current_index, current_factors, cand, running_entropy) + (cand,)

# === Resume Logic ===
def resume_all_checkpoints(n):
    with open("checkpoint_memory.json") as f:
        checkpoints = json.load(f)

    all_perms = list(itertools.permutations(range(1, n + 1)))
    total = len(all_perms)

    for key, chkpt in checkpoints.items():
        print(f"[Resuming from {key}] Step {chkpt['step']} | visited {chkpt['visited_count']} | entropy {chkpt['entropy']:.2f}")

        current = chkpt['path_so_far'][-1]
        current_index = lehmer_encode(current)
        current_factors = factorint(current_index)

        sequence = chkpt['sequence_so_far'][:]
        path = chkpt['path_so_far'][:]
        visited = set(tuple(p) for p in path)

        entropy_history = [UniversalNumber(lehmer_encode(p)).entropy() for p in path[-10:]]
        runtime_window = deque(maxlen=10)
        running_entropy = sum(entropy_history) / len(entropy_history)
        entropy_warnings = 0

        start_time = time.time()
        trace = []

        with get_context("spawn").Pool(processes=cpu_count()) as pool:
            step = chkpt['step']

            while len(visited) < total:
                step_start = time.time()
                remaining = [p for p in all_perms if p not in visited]
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

                trace.append((step, best_index, best_overlap, best_shared, best_entropy, best_norm, step_time_ms, runtime_slope))
                print(f"→ {best_next} | Overlap: {best_overlap} | Step {step} | Entropy: {best_entropy:.4f} | Runtime Slope: {runtime_slope:.2f}")

                # Only warn, don't exit unless coherence fails 5x in a row
                if best_entropy > 1.25 or runtime_slope > 3.5:
                    entropy_warnings += 1
                    if entropy_warnings >= 5:
                        print(f"[Exit @ step {step}] ✘ Coherence degraded for 5 steps — halting resume")
                        break
                else:
                    entropy_warnings = 0

                current = best_next
                current_index = best_index
                current_factors = best_factors
                step += 1

        result_len = len(sequence)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nkey = f"n{n}"

        if os.path.exists("shortest_record.json"):
            with open("shortest_record.json") as rf:
                global_best = json.load(rf)
        else:
            global_best = {}

        best_length = global_best.get(nkey, {}).get("length", float('inf'))

        if result_len < best_length:
            print(f"[✔] New Best Length: {result_len} (old: {best_length})")
            file_name = f"best_resumed_path_{nkey}_{result_len}_{timestamp}.txt"
            trace_name = f"trace_resume_{nkey}_{result_len}_{timestamp}.csv"
            with open(file_name, "w") as out:
                out.write("".join(str(x) for x in sequence))
            with open(trace_name, "w") as tf:
                tf.write("step,index,overlap,shared_primes,entropy,coherence_norm,step_time_ms,runtime_slope\n")
                tf.write("\n".join(",".join(map(str, row)) for row in trace))
            global_best[nkey] = {
                "length": result_len,
                "timestamp": timestamp,
                "path_file": file_name,
                "trace_file": trace_name
            }
            with open("shortest_record.json", "w") as rf:
                json.dump(global_best, rf, indent=2)
        else:
            print(f"[⏹] Resume complete — length {result_len} not better than best ({best_length})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, help="Number of symbols")
    args = parser.parse_args()
    resume_all_checkpoints(args.n)
