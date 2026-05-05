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
# 1. Genome Parsing
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

    # Deduplicate exact duplicates
    unique_v: Dict[Tuple[int, int, int, Tuple[int, int]], Vertex] = {}
    for v in vertices:
        unique_v[(v.color, v.L, v.R, v.adj_key)] = v

    # Reassign vids densely
    out = [
        Vertex(i, v.color, v.L, v.R, v.adj_key)
        for i, v in enumerate(sorted(unique_v.values(), key=lambda x: (x.color, x.L, x.R)))
    ]

    return out


def build_conflict_graph(
    vertices: List[Vertex],
    A: Optional[List[int]] = None,
    *,
    pseudo_l: int = 0,
    pseudo_use_abs: bool = True,
) -> Dict[int, Set[int]]:
    """
    Conflict edges:
      1) interval-overlap edges
      2) same-color clique
      3) pseudo-1 gene-family reuse edges, if pseudo_l == 1

    Touching endpoints are NON-overlapping.
    """
    adj: Dict[int, Set[int]] = {v.vid: set() for v in vertices}

    # 1. Interval overlap conflicts
    sorted_v = sorted(vertices, key=lambda v: (v.L, v.R))
    active: List[Tuple[int, int]] = []
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

    # 2. Same-color clique
    by_color = defaultdict(list)
    for v in vertices:
        by_color[v.color].append(v.vid)

    for group in by_color.values():
        for i in range(len(group)):
            u = group[i]
            for j in range(i + 1, len(group)):
                w = group[j]
                adj[u].add(w)
                adj[w].add(u)

    # 3. pseudo-1 constraint: same gene family cannot be used twice
    if pseudo_l == 1 and A is not None:
        buckets = defaultdict(list)

        for v in vertices:
            a1, a2 = A[v.color], A[v.color + 1]
            g1 = abs(a1) if pseudo_use_abs else a1
            g2 = abs(a2) if pseudo_use_abs else a2

            buckets[g1].append(v.vid)
            buckets[g2].append(v.vid)

        for vids in buckets.values():
            for i in range(len(vids)):
                u = vids[i]
                for j in range(i + 1, len(vids)):
                    w = vids[j]
                    if u != w:
                        adj[u].add(w)
                        adj[w].add(u)

    return adj


def is_pseudo_l_on_adjacencies(
    A: List[int],
    chosen_vertices: List[Vertex],
    ell: int,
    *,
    use_abs: bool = True,
) -> bool:
    """
    Count how many selected adjacencies touch each gene family.
    Require count <= ell.

    pseudo_l=1 means each gene family participates in at most one selected adjacency.
    pseudo_l=2 means each gene family participates in at most two selected adjacencies.
    """
    if ell <= 0:
        return True

    counts: Dict[int, int] = {}

    for v in chosen_vertices:
        a1, a2 = A[v.color], A[v.color + 1]

        g1 = abs(a1) if use_abs else a1
        g2 = abs(a2) if use_abs else a2

        counts[g1] = counts.get(g1, 0) + 1
        if counts[g1] > ell:
            return False

        counts[g2] = counts.get(g2, 0) + 1
        if counts[g2] > ell:
            return False

    return True


# ------------------------------------------------------------------
# 3. MIS Enumeration via Maximal Cliques in Complement
# ------------------------------------------------------------------

def enumerate_maximal_independent_sets_bitset(
    adj: Dict[int, Set[int]],
    all_vids: List[int],
    *,
    time_limit_sec: float = 0.0,
    prefer_large_first: bool = True,
) -> Iterable[Set[int]]:
    start = time.time()
    n = len(all_vids)

    if n == 0:
        return
        yield

    idx_of = {vid: i for i, vid in enumerate(all_vids)}
    vid_of = {i: vid for i, vid in enumerate(all_vids)}

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
        if depth > 20000:
            return

        if not time_ok():
            return

        if P == 0 and X == 0:
            yield {vid_of[i] for i in bits_iter(R)}
            return

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
# 4. Collector for Top-K MIS
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
    H: Optional[List[int]] = None,
    verts: Optional[List[Vertex]] = None,
    pseudo_l: int = 0,
    pseudo_use_abs: bool = True,
    progress_every: int = 0,
    stop_if_no_accept_after: int = 0,
) -> Dict:
    if seed is not None:
        random.seed(seed)

    enumerated = 0
    accepted = 0
    rejects = 0
    start = time.time()

    vmap: Optional[Dict[int, Vertex]] = None
    if pseudo_l > 0:
        if H is None or verts is None:
            raise ValueError("pseudo_l > 0 requires H and verts")
        vmap = {v.vid: v for v in verts}

    def time_ok() -> bool:
        if time_limit_sec <= 0:
            return True
        return (time.time() - start) <= time_limit_sec

    def passes_pseudo(t: Tuple[int, ...]) -> bool:
        if pseudo_l <= 0:
            return True

        assert vmap is not None
        assert H is not None

        chosen = [vmap[vid] for vid in t if vid in vmap]
        return is_pseudo_l_on_adjacencies(
            H,
            chosen,
            pseudo_l,
            use_abs=pseudo_use_abs,
        )

    if store_all_mis:
        all_mis: List[Tuple[int, ...]] = []
        seen: Set[Tuple[int, ...]] = set()

        for mis in enumerate_maximal_independent_sets_bitset(
            adj,
            all_vids,
            time_limit_sec=time_limit_sec,
            prefer_large_first=prefer_large_first,
        ):
            enumerated += 1
            t = tuple(sorted(mis))

            if progress_every > 0 and enumerated % progress_every == 0:
                print(
                    f"      [PROGRESS] enumerated={enumerated}, accepted={accepted}, rejects={rejects}",
                    file=sys.stderr,
                )

            if stop_if_no_accept_after > 0 and accepted == 0 and enumerated >= stop_if_no_accept_after:
                break

            if not passes_pseudo(t):
                rejects += 1
                if max_enumerated > 0 and enumerated >= max_enumerated:
                    break
                if not time_ok():
                    break
                continue

            if t not in seen:
                seen.add(t)
                all_mis.append(t)
                accepted += 1

            if max_enumerated > 0 and enumerated >= max_enumerated:
                break
            if not time_ok():
                break

        all_mis.sort(key=lambda x: (-len(x), x))

        return {
            "top_mis": [list(x) for x in all_mis],
            "top_sizes": [len(x) for x in all_mis],
            "enumerated": enumerated,
            "accepted_after_pseudo_l": accepted,
            "rejects_by_pseudo_l": rejects,
            "accept_rate_over_enumerated": accepted / enumerated if enumerated else 0.0,
            "stored_all_mis": True,
            "is_complete_enumeration": time_limit_sec <= 0.0 and max_enumerated == 0,
            "pseudo_l": pseudo_l,
            "pseudo_use_abs": pseudo_use_abs,
            "note": "All accepted maximal independent sets stored and sorted by size descending.",
        }

    if top_k <= 0:
        return {
            "top_mis": [],
            "top_sizes": [],
            "enumerated": 0,
            "accepted_after_pseudo_l": 0,
            "rejects_by_pseudo_l": 0,
            "pseudo_l": pseudo_l,
            "pseudo_use_abs": pseudo_use_abs,
        }

    heap: List[Tuple[int, Tuple[int, ...]]] = []
    seen: Set[Tuple[int, ...]] = set()

    for mis in enumerate_maximal_independent_sets_bitset(
        adj,
        all_vids,
        time_limit_sec=time_limit_sec,
        prefer_large_first=prefer_large_first,
    ):
        enumerated += 1
        t = tuple(sorted(mis))

        if progress_every > 0 and enumerated % progress_every == 0:
            print(
                f"      [PROGRESS] enumerated={enumerated}, accepted={accepted}, rejects={rejects}, heap={len(heap)}/{top_k}",
                file=sys.stderr,
            )

        if stop_if_no_accept_after > 0 and accepted == 0 and enumerated >= stop_if_no_accept_after:
            break

        if not passes_pseudo(t):
            rejects += 1
            if max_enumerated > 0 and enumerated >= max_enumerated:
                break
            if not time_ok():
                break
            continue

        if t not in seen:
            seen.add(t)
            accepted += 1
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

    is_exact = exact_topk and max_enumerated == 0 and time_limit_sec <= 0.0

    return {
        "top_mis": top_mis,
        "top_sizes": [len(x) for x in top_mis],
        "enumerated": enumerated,
        "accepted_after_pseudo_l": accepted,
        "rejects_by_pseudo_l": rejects,
        "accept_rate_over_enumerated": accepted / enumerated if enumerated else 0.0,
        "stored_all_mis": False,
        "is_exact_topk_among_all": bool(is_exact),
        "pseudo_l": pseudo_l,
        "pseudo_use_abs": pseudo_use_abs,
        "note": (
            "pseudo_l=0 means no pseudo constraint. "
            "pseudo_l=1 is encoded in the graph and also checked. "
            "pseudo_l=2 is checked as an enumeration filter."
        ),
    }


# ------------------------------------------------------------------
# 5. Main Runner
# ------------------------------------------------------------------

def adaptive_time_limit(base_time: float, num_verts: int) -> float:
    if base_time <= 0:
        return 0.0

    if num_verts <= 10_000:
        return min(base_time, 120.0)
    elif num_verts <= 30_000:
        return min(base_time, 180.0)
    else:
        return min(base_time, 120.0)


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
    pseudo_l: int,
    pseudo_use_abs: bool,
    progress_every: int,
    stop_if_no_accept_after: int,
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
            "accepted_after_pseudo_l": 0,
            "rejects_by_pseudo_l": 0,
            "accept_rate_over_enumerated": 0.0,
            "stored_all_mis": store_all_mis,
            "is_exact_topk_among_all": False,
            "signed": signed,
            "minimal_only": minimal_only,
            "pseudo_l": pseudo_l,
            "pseudo_use_abs": pseudo_use_abs,
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
            "accepted_after_pseudo_l": 0,
            "rejects_by_pseudo_l": 0,
            "accept_rate_over_enumerated": 0.0,
            "stored_all_mis": store_all_mis,
            "is_exact_topk_among_all": False,
            "signed": signed,
            "minimal_only": minimal_only,
            "pseudo_l": pseudo_l,
            "pseudo_use_abs": pseudo_use_abs,
        }

    adj = build_conflict_graph(
        verts,
        H,
        pseudo_l=pseudo_l,
        pseudo_use_abs=pseudo_use_abs,
    )

    all_vids = [v.vid for v in verts]
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
        H=H,
        verts=verts,
        pseudo_l=pseudo_l,
        pseudo_use_abs=pseudo_use_abs,
        progress_every=progress_every,
        stop_if_no_accept_after=stop_if_no_accept_after,
    )

    res.update({
        "top_k": top_k,
        "num_vertices": len(verts),
        "num_edges": m2 // 2,
        "signed": signed,
        "minimal_only": minimal_only,
        "effective_time_limit_sec": effective_time,
        "pseudo_l": pseudo_l,
        "pseudo_use_abs": pseudo_use_abs,
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
    pseudo_l: int,
    pseudo_use_abs: bool,
    progress_every: int,
    stop_if_no_accept_after: int,
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
    print(f"[INFO] pseudo_l={pseudo_l}, pseudo_use_abs={pseudo_use_abs}")

    for fp in files:
        f_mode = infer_mode_from_path(fp)
        f_pset = fp.parent.name if fp.parent is not None else "unknown"

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

            out_path = out_dir / f"inst{i + 1}.json"
            if out_path.exists():
                continue

            print(f"  Instance {i + 1} (|A|={len(H)} |B|={len(G)})...")
            print(f"    [INFO] base_time={time_limit_sec}s (adaptive)")

            signed_flag = f_mode == "signed"

            res = process_instance(
                H,
                G,
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
                pseudo_l=pseudo_l,
                pseudo_use_abs=pseudo_use_abs,
                progress_every=progress_every,
                stop_if_no_accept_after=stop_if_no_accept_after,
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
            elif res.get("top_mis"):
                print(
                    f"    Enumerated={res['enumerated']}, "
                    f"Accepted={res.get('accepted_after_pseudo_l', 0)}, "
                    f"Rejects={res.get('rejects_by_pseudo_l', 0)}, "
                    f"Top MIS size={len(res['top_mis'][0])}, K={top_k}"
                )
            else:
                print(
                    f"    Enumerated={res.get('enumerated', 0)}, "
                    f"Accepted={res.get('accepted_after_pseudo_l', 0)}, "
                    f"No MIS produced."
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", default="qingge_datasets")
    parser.add_argument("--out_root", default="mis_outputs_exact_topk")

    parser.add_argument("--only_mode", default=None)
    parser.add_argument("--only_pset", default=None)
    parser.add_argument("--only_file", default=None)
    parser.add_argument("--only_instance", type=int, default=0)

    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--minimal_only", action="store_true")

    parser.add_argument("--max_verts", type=int, default=60000)

    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--max_enumerated", type=int, default=0)
    parser.add_argument("--time_limit", type=float, default=0.0)
    parser.add_argument("--exact_topk", action="store_true")
    parser.add_argument("--store_all_mis", action="store_true")
    parser.add_argument("--prefer_large_first", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--pseudo_l", type=int, default=0,
                        help="0=off, 1=pseudo-1, 2=pseudo-2")

    parser.add_argument("--pseudo_use_abs", action="store_true",
                        help="Use absolute gene family IDs for pseudo-l counting")

    parser.add_argument("--progress_every", type=int, default=0)
    parser.add_argument("--stop_if_no_accept_after", type=int, default=0)

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
        pseudo_l=args.pseudo_l,
        pseudo_use_abs=args.pseudo_use_abs,
        progress_every=args.progress_every,
        stop_if_no_accept_after=args.stop_if_no_accept_after,
    )