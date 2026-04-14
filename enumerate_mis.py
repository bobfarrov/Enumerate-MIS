#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Iterable
import sys
sys.setrecursionlimit(1_000_000)

# ------------------------------------------------------------------
# 1. Genome Parsing (Robust)
# ------------------------------------------------------------------

def parse_genome(s: str) -> List[int]:
    tokens = re.findall(r"[+-]?\d+", s)
    return [int(t) for t in tokens]

# ------------------------------------------------------------------
# 2. Graph & Logic Classes
# ------------------------------------------------------------------

@dataclass(frozen=True)
class Vertex:
    vid: int
    color: int
    L: int
    R: int
    adj_key: Tuple[int, int]

def canonical_signed_adj(a: int, b: int) -> Tuple[int, int]:
    x1, x2 = (a, b), (-b, -a)
    return x1 if x1 <= x2 else x2

def infer_mode_from_path(fp: Path) -> str:
    s = str(fp)
    name = fp.name.lower()
    if "unsigned" in name or "/unsigned/" in s or "\\unsigned\\" in s:
        return "unsigned"
    if "signed" in name or "/signed/" in s or "\\signed\\" in s:
        return "signed"
    return "unknown"

def generate_vertices(A: List[int], B: List[int], signed: bool, minimal_only: bool) -> List[Vertex]:
    """
    NOTE: This keeps your original modeling:
      - pos[] built from B
      - colors are 'c' = adjacency index in A (c=0..|A|-2)
      - vertices are intervals (L,R) over indices in B
    """
    pos = defaultdict(list)
    for i, x in enumerate(B):
        k = x if signed else abs(x)
        pos[k].append(i)

    vertices: List[Vertex] = []
    vid = 0
    for c in range(len(A) - 1):
        a1, a2 = A[c], A[c + 1]
        if signed:
            adj_key = canonical_signed_adj(a1, a2)
            pairs = {(adj_key[0], adj_key[1]), (-adj_key[1], -adj_key[0])}
        else:
            adj_key = tuple(sorted((abs(a1), abs(a2))))
            pairs = {(abs(a1), abs(a2)), (abs(a2), abs(a1))}

        candidates: List[Tuple[int, int]] = []
        for u, v in pairs:
            Pu, Pv = pos[u], pos[v]
            if minimal_only:
                for i in Pu:
                    best_j = None
                    min_dist = float("inf")
                    for j in Pv:
                        if i == j:
                            continue
                        dist = abs(i - j)
                        if dist < min_dist:
                            min_dist = dist
                            best_j = j
                    if best_j is not None:
                        L, R = sorted((i, best_j))
                        candidates.append((L, R))
            else:
                for i in Pu:
                    for j in Pv:
                        if i == j:
                            continue
                        L, R = (i, j) if i < j else (j, i)
                        candidates.append((L, R))

        for L, R in candidates:
            vertices.append(Vertex(vid, c, L, R, adj_key))
            vid += 1

    # De-duplicate exact duplicates
    unique_v: Dict[Tuple[int, int, int, Tuple[int, int]], Vertex] = {}
    for v in vertices:
        unique_v[(v.color, v.L, v.R, v.adj_key)] = v

    # Re-assign vids densely
    out = [Vertex(i, v.color, v.L, v.R, v.adj_key)
           for i, v in enumerate(sorted(unique_v.values(), key=lambda x: (x.color, x.L, x.R)))]
    return out

def build_conflict_graph(vertices: List[Vertex]) -> Dict[int, Set[int]]:
    """
    Conflict edges:
      1) interval-overlap edges across different colors
      2) within the same color: clique (at most one per color)

    Overlap rule used here treats touching endpoints as NON-overlapping (since we pop old with R <= L).
    If you want endpoints to conflict, change `<=` to `<`.
    """
    adj: Dict[int, Set[int]] = {v.vid: set() for v in vertices}
    sorted_v = sorted(vertices, key=lambda v: (v.L, v.R))

    active: List[Tuple[int, int]] = []  # (R, vid)
    active_set: Set[int] = set()

    for v in sorted_v:
        while active and active[0][0] <= v.L:
            _, old_id = heapq.heappop(active)
            active_set.discard(old_id)

        for other_id in active_set:
            adj[v.vid].add(other_id)
            adj[other_id].add(v.vid)

        heapq.heappush(active, (v.R, v.vid))
        active_set.add(v.vid)

    by_color = defaultdict(list)
    for v in vertices:
        by_color[v.color].append(v.vid)
    for group in by_color.values():
        # clique inside each color
        for i in range(len(group)):
            u = group[i]
            for j in range(i + 1, len(group)):
                w = group[j]
                adj[u].add(w)
                adj[w].add(u)

    return adj

# ------------------------------------------------------------------
# 3. Exact MIS Enumeration via Maximal Cliques in Complement (Bitset BK)
# ------------------------------------------------------------------

def enumerate_maximal_independent_sets_bitset(
    adj: Dict[int, Set[int]],
    all_vids: List[int],
    *,
    time_limit_sec: float = 0.0,
    prefer_large_first: bool = True
) -> Iterable[Set[int]]:
    """
    Enumerate ALL maximal independent sets exactly.
    Implementation: enumerate maximal cliques in complement graph via Bron–Kerbosch with pivot,
    using Python int bitsets.

    time_limit_sec:
      - 0 => no limit
      - >0 => stop yielding after time limit (still exact for those yielded, but incomplete overall)

    prefer_large_first:
      - True => heuristic ordering to tend to produce larger MIS earlier (NOT a guarantee).
    """
    start = time.time()

    n = len(all_vids)
    if n == 0:
        return
        yield  # for type checkers

    idx_of = {vid: i for i, vid in enumerate(all_vids)}
    vid_of = {i: vid for i, vid in enumerate(all_vids)}

    # adjacency bitmasks in original graph
    adj_mask = [0] * n
    for vid in all_vids:
        i = idx_of[vid]
        m = 0
        for nb in adj[vid]:
            j = idx_of.get(nb)
            if j is not None:
                m |= (1 << j)
        adj_mask[i] = m

    all_mask = (1 << n) - 1

    # complement neighbors (exclude self)
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = all_mask & ~(adj_mask[i] | (1 << i))

    def bits_iter(x: int):
        while x:
            lsb = x & -x
            i = lsb.bit_length() - 1
            yield i
            x ^= lsb

    def popcount(x: int) -> int:
        return x.bit_count()

    def time_ok() -> bool:
        if time_limit_sec <= 0:
            return True
        return (time.time() - start) <= time_limit_sec

    def bronk(R: int, P: int, X: int, depth: int = 0):
        # Safety guard against runaway recursion
        if depth > 20000:
            return

        if not time_ok():
            return

        if P == 0 and X == 0:
            yield {vid_of[i] for i in bits_iter(R)}
            return

        # Choose pivot u from P|X maximizing |P ∩ N(u)|
        PX = P | X
        u = None
        max_deg = -1
        tmp = PX
        while tmp:
            lsb = tmp & -tmp
            ui = lsb.bit_length() - 1
            deg = popcount(P & comp_mask[ui])
            if deg > max_deg:
                max_deg = deg
                u = ui
            tmp ^= lsb

        # Candidates: P \ N(u)
        candidates = P & ~comp_mask[u] if u is not None else P

        cand_list = list(bits_iter(candidates))
        if prefer_large_first and 1 < len(cand_list) <= 2000:
            cand_list.sort(key=lambda v: popcount(P & comp_mask[v]), reverse=True)

        for v in cand_list:
            Nv = comp_mask[v]
            yield from bronk(R | (1 << v), P & Nv, X & Nv, depth + 1)
            P &= ~(1 << v)
            X |= (1 << v)
            if not time_ok():
                return

    yield from bronk(0, all_mask, 0, 0)

# ------------------------------------------------------------------
# 4. Collector for maximal IS
# ------------------------------------------------------------------

def collect_top_k_mis(
    adj: Dict[int, Set[int]],
    all_vids: List[int],
    *,
    top_k: int,
    max_enumerated: int = 0,
    time_limit_sec: float = 0.0,
    exact_topk: bool = False,
    prefer_large_first: bool = True,
    seed: Optional[int] = None,
    store_all_mis: bool = False,
) -> Dict:
    """
    If store_all_mis=False:
        keep only Top-K as before.

    If store_all_mis=True:
        enumerate all MIS, store all of them, sort by size descending.
    """
    if seed is not None:
        random.seed(seed)

    enumerated = 0
    start = time.time()

    def time_ok() -> bool:
        if time_limit_sec <= 0:
            return True
        return (time.time() - start) <= time_limit_sec

    if store_all_mis:
        all_mis: List[Tuple[int, ...]] = []
        seen: Set[Tuple[int, ...]] = set()

        for mis in enumerate_maximal_independent_sets_bitset(
            adj, all_vids, time_limit_sec=time_limit_sec, prefer_large_first=prefer_large_first
        ):
            enumerated += 1
            t = tuple(sorted(mis))
            if t not in seen:
                seen.add(t)
                all_mis.append(t)
            
            if enumerated % 100 == 0:
                print(f"      [PROGRESS] enumerated={enumerated}, stored={len(all_mis)}")

            if max_enumerated > 0 and enumerated >= max_enumerated:
                break
            if not time_ok():
                break

        all_mis.sort(key=lambda x: (-len(x), x))

        return {
            "top_mis": [list(x) for x in all_mis],
            "top_sizes": [len(x) for x in all_mis],
            "enumerated": enumerated,
            "stored_all_mis": True,
            "is_complete_enumeration": (time_limit_sec <= 0.0 and max_enumerated == 0),
            "note": "All maximal independent sets stored and sorted by size descending."
        }

    if top_k <= 0:
        return {"top_mis": [], "enumerated": 0, "is_exact": False}

    heap: List[Tuple[int, Tuple[int, ...]]] = []
    seen: Set[Tuple[int, ...]] = set()

    for mis in enumerate_maximal_independent_sets_bitset(
        adj, all_vids, time_limit_sec=time_limit_sec, prefer_large_first=prefer_large_first
    ):
        enumerated += 1
        t = tuple(sorted(mis))
        if t not in seen:
            seen.add(t)
            s = len(t)
            if len(heap) < top_k:
                heapq.heappush(heap, (s, t))
            else:
                if s > heap[0][0]:
                    heapq.heapreplace(heap, (s, t))

        if not exact_topk and len(heap) >= top_k:
            break
        if max_enumerated > 0 and enumerated >= max_enumerated:
            break
        if not time_ok():
            break

    heap.sort(key=lambda x: x[0], reverse=True)
    top_mis = [list(t) for _, t in heap]

    is_exact = exact_topk and (max_enumerated == 0) and (time_limit_sec <= 0.0)
    return {
        "top_mis": top_mis,
        "top_sizes": [len(x) for x in top_mis],
        "enumerated": enumerated,
        "stored_all_mis": False,
        "is_exact_topk_among_all": bool(is_exact),
        "note": (
            "If store_all_mis is False, only Top-K are stored. "
            "If exact_topk is False, Top-K is heuristic unless full enumeration is completed."
        ),
    }

# ------------------------------------------------------------------
# 5. Main Runner
# ------------------------------------------------------------------

def adaptive_time_limit(base_time: float, num_verts: int) -> float:
    """
    Returns an effective per-instance time budget based on num_verts.
    - If base_time <= 0: no time limit (0 means unlimited in your code)
    - Otherwise: cap by tiers.
    """
    if base_time <= 0:
        return 0.0  # unlimited

    if num_verts <= 10_000:
        return min(base_time, 60.0)
    elif num_verts <= 30_000:
        return min(base_time, 30.0)
    else:
        return min(base_time, 10.0)

def process_instance(
    H: List[int],
    G: List[int],
    signed: bool,
    minimal_only: bool,
    top_k: int,
    max_enumerated: int,
    time_limit_sec: float,
    exact_topk: bool,
    prefer_large_first: bool,
    seed: Optional[int],
    max_verts: int,
    store_all_mis: bool,
) -> Dict:
    verts = generate_vertices(H, G, signed, minimal_only)
    effective_time = adaptive_time_limit(time_limit_sec, len(verts))

    if len(verts) > max_verts:
        return {
            "skipped": True,
            "reason": f"too many vertices: {len(verts)} > {max_verts}",
            "top_k": top_k,
            "num_vertices": len(verts),
            "num_edges": 0,
            "top_mis": [],
            "top_sizes": [],
            "enumerated": 0,
            "stored_all_mis": store_all_mis,
            "is_exact_topk_among_all": False,
            "signed": signed,
            "minimal_only": minimal_only,
        }

    if not verts:
        return {
            "skipped": False,
            "top_k": top_k,
            "num_vertices": 0,
            "num_edges": 0,
            "top_mis": [],
            "top_sizes": [],
            "enumerated": 0,
            "stored_all_mis": store_all_mis,
            "is_exact_topk_among_all": False,
        }

    adj = build_conflict_graph(verts)
    all_vids = [v.vid for v in verts]

    # edge count for reporting
    m2 = sum(len(adj[v]) for v in adj)

    res = collect_top_k_mis(
        adj,
        all_vids,
        top_k=top_k,
        max_enumerated=max_enumerated,
        time_limit_sec=effective_time,
        exact_topk=exact_topk,
        prefer_large_first=prefer_large_first,
        seed=seed,
        store_all_mis=store_all_mis,
    )

    res.update({
        "top_k": top_k,
        "num_vertices": len(verts),
        "num_edges": m2 // 2,
        "signed": signed,
        "minimal_only": minimal_only,
        "effective_time_limit_sec": effective_time,
    })
    return res

def run_dataset(
    root: str,
    out_root: str,
    only_mode: Optional[str],
    only_pset: Optional[str],
    only_file: Optional[str],
    only_instance: int,
    max_files: int,
    minimal_only: bool,
    top_k: int,
    max_enumerated: int,
    time_limit_sec: float,
    exact_topk: bool,
    prefer_large_first: bool,
    seed: Optional[int],
    max_verts: int,
    store_all_mis: bool,
):
    files = sorted(Path(root).rglob("*.txt"))

    if only_mode:
        files = [f for f in files if infer_mode_from_path(f) == only_mode]
    if only_pset:
        files = [f for f in files if only_pset in str(f)]
    if only_file:
        files = [f for f in files if f.name == only_file]

    if max_files > 0:
        files = files[:max_files]

    print(f"Found {len(files)} files to process.")

    for fp in files:
        try:
            f_mode = infer_mode_from_path(fp)
            f_pset = fp.parent.name if fp.parent is not None else "unknown"
        except Exception:
            f_mode, f_pset = "unknown", "unknown"

        print(f"Processing: {fp}")
        instances: List[Tuple[List[int], List[int]]] = []

        with open(fp, "r", encoding="utf-8-sig") as f:
            H_curr = None
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                if line.upper().startswith("H:"):
                    H_curr = parse_genome(line)
                elif line.upper().startswith("G:"):
                    if H_curr:
                        G_curr = parse_genome(line)
                        instances.append((H_curr, G_curr))
                        H_curr = None
                    else:
                        print(f"  [WARNING] Line {line_num}: Found G but no matching H.")

        if not instances:
            print(f"  [ERROR] No instances found in {fp}. Check file format.")
            continue

        out_dir = Path(out_root) / f_mode / f_pset / fp.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, (H, G) in enumerate(instances):
            if only_instance > 0 and (i + 1) != only_instance:
                continue

            out_path = out_dir / f"inst{i+1}.json"
            if out_path.exists():
                continue

            print(f"  Instance {i+1} (|A|={len(H)} |B|={len(G)})...")
            print(f"    [INFO] base_time={time_limit_sec}s (adaptive)")

            signed_flag = (f_mode == "signed")
            res = process_instance(
                H, G,
                signed=signed_flag,
                minimal_only=minimal_only,
                top_k=top_k,
                max_enumerated=max_enumerated,
                time_limit_sec=time_limit_sec,
                exact_topk=exact_topk,
                prefer_large_first=prefer_large_first,
                seed=seed,
                max_verts=max_verts,
                store_all_mis=store_all_mis,
            )
            res["instance_id"] = i + 1
            res["source_file"] = str(fp)

            if "effective_time_limit_sec" in res:
                print(
                    f"    [INFO] effective_time={res['effective_time_limit_sec']}s, "
                    f"verts={res.get('num_vertices')}"
                )

            with open(out_path, "w") as f_out:
                json.dump(res, f_out, indent=2)

            if res.get("skipped"):
                print(f"    SKIPPED: {res['reason']}")
            elif res["top_mis"]:
                if res.get("stored_all_mis", False):
                    print(f"    Enumerated ALL MIS={res['enumerated']}, Largest MIS size={len(res['top_mis'][0])}")
                else:
                    print(f"    Enumerated={res['enumerated']}, Top MIS size={len(res['top_mis'][0])}, K={top_k}")
            else:
                print(f"    Enumerated={res['enumerated']}, No MIS produced (maybe time/limit too small).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", default="qingge_datasets")
    parser.add_argument("--out_root", default="mis_outputs_exact_topk")

    parser.add_argument("--only_mode", default=None)   # "signed" or "unsigned"
    parser.add_argument("--only_pset", default=None)
    parser.add_argument("--only_file", default=None,
                        help="Process only this exact filename")
    parser.add_argument("--only_instance", type=int, default=0,
                        help="Process only this 1-based instance id inside the file (0 = all)")

    parser.add_argument("--max_files", type=int, default=0)

    parser.add_argument("--minimal_only", action="store_true")

    parser.add_argument("--max_verts", type=int, default=60000,
                        help="Skip an instance if generated vertices exceed this threshold")

    # Top-K / all-MIS options
    parser.add_argument("--top_k", type=int, default=30, help="How many maximal IS to keep")
    parser.add_argument("--max_enumerated", type=int, default=0,
                        help="Stop after enumerating this many MIS (0 = no limit)")
    parser.add_argument("--time_limit", type=float, default=0.0,
                        help="Stop after this many seconds (0 = no limit)")
    parser.add_argument("--exact_topk", action="store_true",
                        help="If set: keep true Top-K among enumerated MIS (exact only if no caps/time limit). "
                             "If not set: stop early after collecting K (fast heuristic).")
    parser.add_argument("--store_all_mis", action="store_true",
                        help="Store all maximal independent sets, sorted by size descending")

    parser.add_argument("--prefer_large_first", action="store_true",
                        help="Heuristic ordering to output larger MIS earlier (recommended)")

    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    run_dataset(
        root=args.dataset_root,
        out_root=args.out_root,
        only_mode=args.only_mode,
        only_pset=args.only_pset,
        only_file=args.only_file,
        only_instance=args.only_instance,
        max_files=args.max_files,
        minimal_only=args.minimal_only,
        top_k=args.top_k,
        max_enumerated=args.max_enumerated,
        time_limit_sec=args.time_limit,
        exact_topk=args.exact_topk,
        prefer_large_first=args.prefer_large_first,
        seed=args.seed,
        max_verts=args.max_verts,
        store_all_mis=args.store_all_mis,
    )