#!/usr/bin/env python3
from __future__ import annotations

import os
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from multiprocessing import Pool, cpu_count
import time


# =========================================================
# 9 mutation operations (Qingge TCBB generator style)
# =========================================================

def reverse_segment_signed(seg: List[int], signed: bool) -> List[int]:
    seg2 = list(reversed(seg))
    if signed:
        seg2 = [-x for x in seg2]
    return seg2

def unit_reversal_swap(G: List[int], l: int, signed: bool) -> None:
    # swap g_k and g_{k+l}
    if len(G) <= l:
        return
    k = random.randrange(0, len(G) - l)
    i, j = k, k + l
    G[i], G[j] = G[j], G[i]

def unit_insertion_new(G: List[int], next_new_gene: int, signed: bool) -> int:
    # insert a new gene family
    k = random.randrange(0, len(G) + 1)
    gene = next_new_gene
    if signed and random.random() < 0.5:
        gene = -gene
    G.insert(k, gene)
    return next_new_gene + 1

def unit_deletion(G: List[int]) -> None:
    if not G:
        return
    k = random.randrange(0, len(G))
    G.pop(k)

def unit_duplication(G: List[int]) -> None:
    # copy g_k and insert at k+1
    if not G:
        return
    k = random.randrange(0, len(G))
    G.insert(k + 1, G[k])

def segment_reversal(G: List[int], l: int, signed: bool) -> None:
    # reverse ordering between g_k and g_{k+l} (segment length l+1)
    if len(G) <= l:
        return
    k = random.randrange(0, len(G) - l)
    seg = G[k:k + l + 1]
    G[k:k + l + 1] = reverse_segment_signed(seg, signed)

def tandem_duplication(G: List[int], l: int) -> None:
    # copy segment [k..k+l] and insert at k+l+1
    if len(G) <= l:
        return
    k = random.randrange(0, len(G) - l)
    seg = G[k:k + l + 1]
    G[k + l + 1:k + l + 1] = seg

def segment_deletion(G: List[int], l: int) -> None:
    # delete segment [k..k+l]
    if len(G) <= l:
        return
    k = random.randrange(0, len(G) - l)
    del G[k:k + l + 1]

def segment_duplication_elsewhere(G: List[int], l: int) -> None:
    # copy segment [k..k+l] and insert at a random location outside that region
    if len(G) <= l:
        return
    k = random.randrange(0, len(G) - l)
    seg = G[k:k + l + 1]
    disallowed_start = k
    disallowed_end = k + l  # approximate
    candidates = [i for i in range(0, len(G) + 1) if not (disallowed_start <= i <= disallowed_end)]
    if not candidates:
        return
    ins = random.choice(candidates)
    G[ins:ins] = seg

def transposition(G: List[int], l: int) -> None:
    # cut segment of length l and insert elsewhere (move)
    if len(G) < l + 1 or l <= 0:
        return
    k = random.randrange(0, len(G) - l + 1)
    seg = G[k:k + l]
    del G[k:k + l]
    ins = random.randrange(0, len(G) + 1)
    G[ins:ins] = seg


# =========================================================
# Params P1..P4 (from the paper)
# =========================================================

@dataclass(frozen=True)
class Params:
    p1: float; p2: float; p3: float; p4: float; p5: float
    p6: float; p7: float; p8: float; p9: float
    l: int

PSETS: Dict[str, Params] = {
    "P1": Params(0.05, 0.10, 0.05, 0.05, 0.03, 0.06, 0.03, 0.10, 0.07, 5),
    "P2": Params(0.20, 0.15, 0.15, 0.10, 0.05, 0.08, 0.04, 0.12, 0.10, 1),
    "P3": Params(0.20, 0.18, 0.10, 0.10, 0.05, 0.09, 0.05, 0.10, 0.00, 10),
    "P4": Params(0.20, 0.18, 0.10, 0.10, 0.05, 0.09, 0.05, 0.10, 0.00, 5),
}

N_GRID = [500, 1000, 3000, 5000, 7000, 9000, 12000]
M_GRID = [1, 3, 5]
TRIES = 10


def weighted_choice(ops: List[Tuple[str, float]]) -> str:
    r = random.random()
    s = 0.0
    for name, p in ops:
        s += p
        if r <= s:
            return name
    return ops[-1][0]


def simulate_instance(n: int, m: int, params: Params, signed: bool, seed: int) -> Tuple[List[int], List[int]]:
    random.seed(seed)

    H = list(range(1, n + 1))
    if signed:
        H = [x if random.random() < 0.5 else -x for x in H]

    G = H[:]
    next_new_gene = n + 1

    probs = [
        ("p1_unit_reversal_swap", params.p1),
        ("p2_unit_insertion", params.p2),
        ("p3_unit_deletion", params.p3),
        ("p4_unit_duplication", params.p4),
        ("p5_segment_reversal", params.p5),
        ("p6_tandem_duplication", params.p6),
        ("p7_segment_deletion", params.p7),
        ("p8_segment_dup_elsewhere", params.p8),
        ("p9_transposition", params.p9),
    ]
    total = sum(p for _, p in probs)
    probs = [(name, p / total) for name, p in probs]

    for _cycle in range(m):
        steps = max(1, len(G))  # snapshot length
        for _ in range(steps):
            op = weighted_choice(probs)

            if op == "p1_unit_reversal_swap":
                unit_reversal_swap(G, params.l, signed)
            elif op == "p2_unit_insertion":
                next_new_gene = unit_insertion_new(G, next_new_gene, signed)
            elif op == "p3_unit_deletion":
                unit_deletion(G)
            elif op == "p4_unit_duplication":
                unit_duplication(G)
            elif op == "p5_segment_reversal":
                segment_reversal(G, params.l, signed)
            elif op == "p6_tandem_duplication":
                tandem_duplication(G, params.l)
            elif op == "p7_segment_deletion":
                segment_deletion(G, params.l)
            elif op == "p8_segment_dup_elsewhere":
                segment_duplication_elsewhere(G, params.l)
            elif op == "p9_transposition":
                transposition(G, params.l)
            else:
                raise RuntimeError(op)

    return H, G


# =========================================================
# Crash-safe resume helpers
# =========================================================

def count_instances_in_file(path: str) -> int:
    """Counts how many '# instance' blocks exist in an output file."""
    if not os.path.exists(path):
        return 0
    cnt = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# instance"):
                cnt += 1
    return cnt

def append_instances(path: str, instances: List[Tuple[List[int], List[int]]], start_index: int) -> None:
    """Append instances starting from given instance index (1-based labeling)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for idx, (H, G) in enumerate(instances, start_index):
            f.write(f"# instance {idx}\n")
            f.write("H: " + " ".join(map(str, H)) + "\n")
            f.write("G: " + " ".join(map(str, G)) + "\n\n")

def write_instances_atomic(path: str, instances: List[Tuple[List[int], List[int]]]) -> None:
    """Write full file via temp + atomic rename."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for idx, (H, G) in enumerate(instances, 1):
            f.write(f"# instance {idx}\n")
            f.write("H: " + " ".join(map(str, H)) + "\n")
            f.write("G: " + " ".join(map(str, G)) + "\n\n")
    os.replace(tmp, path)


# =========================================================
# Parallel job
# =========================================================

def job_generate_one_file(args: Tuple[str, str, int, int, bool, str, int]) -> str:
    """
    One job generates (or resumes) ONE dataset file:
      (mode_name, pset_name, n, m, signed, outpath, base_seed)
    """
    mode_name, pset_name, n, m, signed, outpath, base_seed = args
    params = PSETS[pset_name]

    done = count_instances_in_file(outpath)
    if done >= TRIES:
        return f"SKIP {outpath}"

    # Generate missing instances only
    instances: List[Tuple[List[int], List[int]]] = []
    for t in range(done, TRIES):
        seed = base_seed + (hash((mode_name, pset_name, n, m, t)) & 0x7fffffff)
        H, G = simulate_instance(n=n, m=m, params=params, signed=signed, seed=seed)
        instances.append((H, G))

    if done == 0:
        # write all 10 via atomic write
        # but we only have missing list; re-generate full 10 for atomic simplicity
        full: List[Tuple[List[int], List[int]]] = []
        for t in range(TRIES):
            seed = base_seed + (hash((mode_name, pset_name, n, m, t)) & 0x7fffffff)
            full.append(simulate_instance(n=n, m=m, params=params, signed=signed, seed=seed))
        write_instances_atomic(outpath, full)
        return f"WRITE {outpath}"
    else:
        # append remaining safely
        append_instances(outpath, instances, start_index=done + 1)
        return f"APPEND {outpath}"


# =========================================================
# Simple progress bar (no external libs)
# =========================================================

def print_progress(done: int, total: int, start_time: float) -> None:
    width = 30
    frac = done / total if total else 1.0
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else float("inf")
    eta_str = f"{eta:,.0f}s" if eta != float("inf") else "?"
    print(f"\r[{bar}] {done}/{total}  elapsed={elapsed:,.0f}s  eta={eta_str}", end="", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="qingge_datasets", help="Output directory")
    ap.add_argument("--base_seed", type=int, default=0, help="Base RNG seed")
    ap.add_argument("--signed", action="store_true", help="Generate signed datasets")
    ap.add_argument("--unsigned", action="store_true", help="Generate unsigned datasets")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1), help="Parallel workers")
    args = ap.parse_args()

    # Which modes?
    if not args.signed and not args.unsigned:
        modes = [("unsigned", False), ("signed", True)]
    else:
        modes = []
        if args.unsigned:
            modes.append(("unsigned", False))
        if args.signed:
            modes.append(("signed", True))

    # Build job list (one job per output file)
    jobs: List[Tuple[str, str, int, int, bool, str, int]] = []
    for mode_name, signed in modes:
        for pset_name in sorted(PSETS.keys()):
            for n in N_GRID:
                for m in M_GRID:
                    filename = f"{pset_name}_n{n}_m{m}_{mode_name}_tries{TRIES}.txt"
                    outpath = os.path.join(args.outdir, mode_name, pset_name, filename)
                    jobs.append((mode_name, pset_name, n, m, signed, outpath, args.base_seed))

    total = len(jobs)
    start = time.time()
    done = 0
    print(f"Total files to generate/check: {total}")
    print(f"Workers: {args.workers}")
    print("Progress:")

    with Pool(processes=args.workers) as pool:
        for msg in pool.imap_unordered(job_generate_one_file, jobs):
            done += 1
            # optional: print msg sometimes
            # print("\n", msg)
            print_progress(done, total, start)

    print("\nDone.")


if __name__ == "__main__":
    main()
