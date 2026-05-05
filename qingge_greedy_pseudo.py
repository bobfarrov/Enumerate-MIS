#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple


def parse_genome(s: str) -> List[int]:
    tokens = re.findall(r"[+-]?\d+", s)
    return [int(t) for t in tokens]


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
    pos: DefaultDict[int, List[int]] = defaultdict(list)
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

    unique_v: Dict[Tuple[int, int, int, Tuple[int, int]], Vertex] = {}
    for v in vertices:
        unique_v[(v.color, v.L, v.R, v.adj_key)] = v

    out = [
        Vertex(i, v.color, v.L, v.R, v.adj_key)
        for i, v in enumerate(sorted(unique_v.values(), key=lambda x: (x.color, x.L, x.R)))
    ]

    return out


def intervals_overlap_open(v1: Vertex, v2: Vertex) -> bool:
    return not (v1.R <= v2.L or v2.R <= v1.L)


def gene_keys_for_vertex(H: List[int], v: Vertex, pseudo_use_abs: bool) -> Tuple[int, int]:
    a1, a2 = H[v.color], H[v.color + 1]
    if pseudo_use_abs:
        return abs(a1), abs(a2)
    return a1, a2


def greedy_miscig(
    vertices: List[Vertex],
    H: List[int],
    *,
    pseudo_l: int = 0,
    pseudo_use_abs: bool = True,
) -> List[Vertex]:
    """
    Greedy MIS-CIG with optional pseudo-l constraint.

    pseudo_l=0:
        original greedy MIS-CIG

    pseudo_l=1:
        each gene family can appear in at most one selected adjacency

    pseudo_l=2:
        each gene family can appear in at most two selected adjacencies
    """
    alive = sorted(vertices, key=lambda v: (v.R, v.L, v.color, v.vid))
    solution: List[Vertex] = []

    gene_count: Dict[int, int] = defaultdict(int)

    while alive:
        chosen = None

        # find first interval that does not violate pseudo-l
        for candidate in alive:
            if pseudo_l <= 0:
                chosen = candidate
                break

            g1, g2 = gene_keys_for_vertex(H, candidate, pseudo_use_abs)
            if gene_count[g1] + 1 <= pseudo_l and gene_count[g2] + 1 <= pseudo_l:
                chosen = candidate
                break

        if chosen is None:
            break

        solution.append(chosen)

        if pseudo_l > 0:
            g1, g2 = gene_keys_for_vertex(H, chosen, pseudo_use_abs)
            gene_count[g1] += 1
            gene_count[g2] += 1

        nxt: List[Vertex] = []
        for v in alive:
            if v.vid == chosen.vid:
                continue

            same_color = v.color == chosen.color
            overlap = intervals_overlap_open(v, chosen)

            if same_color or overlap:
                continue

            # For pseudo-l, remove intervals that would violate gene count
            if pseudo_l > 0:
                g1, g2 = gene_keys_for_vertex(H, v, pseudo_use_abs)
                if gene_count[g1] + 1 > pseudo_l or gene_count[g2] + 1 > pseudo_l:
                    continue

            nxt.append(v)

        alive = nxt

    return solution


def is_independent_set(vertices: List[Vertex], picked_vids: Iterable[int]) -> bool:
    id_to_v = {v.vid: v for v in vertices}
    picked = [id_to_v[x] for x in picked_vids]

    seen_colors: Set[int] = set()
    for v in picked:
        if v.color in seen_colors:
            return False
        seen_colors.add(v.color)

    for i in range(len(picked)):
        for j in range(i + 1, len(picked)):
            if intervals_overlap_open(picked[i], picked[j]):
                return False

    return True


def satisfies_pseudo_l(
    H: List[int],
    chosen: List[Vertex],
    *,
    pseudo_l: int,
    pseudo_use_abs: bool,
) -> bool:
    if pseudo_l <= 0:
        return True

    counts: Dict[int, int] = defaultdict(int)

    for v in chosen:
        g1, g2 = gene_keys_for_vertex(H, v, pseudo_use_abs)
        counts[g1] += 1
        counts[g2] += 1

        if counts[g1] > pseudo_l or counts[g2] > pseudo_l:
            return False

    return True


def build_conflict_graph(vertices: List[Vertex]) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = {v.vid: set() for v in vertices}

    sorted_v = sorted(vertices, key=lambda v: (v.L, v.R, v.color, v.vid))
    for i in range(len(sorted_v)):
        vi = sorted_v[i]
        for j in range(i + 1, len(sorted_v)):
            vj = sorted_v[j]
            if vj.L >= vi.R:
                break
            if vi.color != vj.color and intervals_overlap_open(vi, vj):
                adj[vi.vid].add(vj.vid)
                adj[vj.vid].add(vi.vid)

    by_color: DefaultDict[int, List[int]] = defaultdict(list)
    for v in vertices:
        by_color[v.color].append(v.vid)

    for group in by_color.values():
        for i in range(len(group)):
            u = group[i]
            for j in range(i + 1, len(group)):
                w = group[j]
                adj[u].add(w)
                adj[w].add(u)

    return adj


def summarize_solution(
    H: List[int],
    vertices: List[Vertex],
    chosen: List[Vertex],
    *,
    pseudo_l: int,
    pseudo_use_abs: bool,
) -> Dict:
    by_color: DefaultDict[int, List[int]] = defaultdict(list)
    for v in chosen:
        by_color[v.color].append(v.vid)

    return {
        "size": len(chosen),
        "vertex_ids": [v.vid for v in chosen],
        "colors": [v.color for v in chosen],
        "unique_colors": len(by_color),
        "is_independent": is_independent_set(vertices, [v.vid for v in chosen]),
        "satisfies_pseudo_l": satisfies_pseudo_l(
            H,
            chosen,
            pseudo_l=pseudo_l,
            pseudo_use_abs=pseudo_use_abs,
        ),
        "pseudo_l": pseudo_l,
        "pseudo_use_abs": pseudo_use_abs,
        "intervals": [asdict(v) for v in chosen],
    }


def process_instance(
    H: List[int],
    G: List[int],
    *,
    signed: bool,
    minimal_only: bool,
    pseudo_l: int,
    pseudo_use_abs: bool,
) -> Dict:
    vertices = generate_vertices(H, G, signed=signed, minimal_only=minimal_only)

    greedy_sol = greedy_miscig(
        vertices,
        H,
        pseudo_l=pseudo_l,
        pseudo_use_abs=pseudo_use_abs,
    )

    adj = build_conflict_graph(vertices)
    num_edges = sum(len(x) for x in adj.values()) // 2

    return {
        "signed": signed,
        "minimal_only": minimal_only,
        "pseudo_l": pseudo_l,
        "pseudo_use_abs": pseudo_use_abs,
        "num_vertices": len(vertices),
        "num_edges": num_edges,
        "greedy_solution": summarize_solution(
            H,
            vertices,
            greedy_sol,
            pseudo_l=pseudo_l,
            pseudo_use_abs=pseudo_use_abs,
        ),
    }


def load_instances_from_txt(fp: Path) -> List[Tuple[List[int], List[int]]]:
    instances: List[Tuple[List[int], List[int]]] = []
    with open(fp, "r", encoding="utf-8-sig") as f:
        H_curr: Optional[List[int]] = None

        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            if line.upper().startswith("H:"):
                H_curr = parse_genome(line)

            elif line.upper().startswith("G:"):
                if H_curr is None:
                    raise ValueError(f"Line {line_num}: found G without matching H in {fp}")
                G_curr = parse_genome(line)
                instances.append((H_curr, G_curr))
                H_curr = None

    return instances


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy MIS-CIG baseline with pseudo-l support")
    parser.add_argument("--dataset_root", default="qingge_datasets")
    parser.add_argument("--out_root", default="qingge_greedy_outputs")
    parser.add_argument("--only_mode", default=None, help="signed or unsigned")
    parser.add_argument("--only_pset", default=None)
    parser.add_argument("--only_file", default=None)
    parser.add_argument("--only_instance", type=int, default=0)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--minimal_only", action="store_true")

    parser.add_argument("--pseudo_l", type=int, default=0,
                        help="0=off, 1=pseudo-1, 2=pseudo-2")

    parser.add_argument("--pseudo_use_abs", action="store_true",
                        help="Count signed genes by absolute gene family")

    args = parser.parse_args()

    files = sorted(Path(args.dataset_root).rglob("*.txt"))

    if args.only_mode:
        files = [f for f in files if infer_mode_from_path(f) == args.only_mode]
    if args.only_pset:
        files = [f for f in files if args.only_pset in str(f)]
    if args.only_file:
        files = [f for f in files if f.name == args.only_file]
    if args.max_files > 0:
        files = files[:args.max_files]

    print(f"Found {len(files)} files to process.")
    print(f"[INFO] pseudo_l={args.pseudo_l}, pseudo_use_abs={args.pseudo_use_abs}")

    for fp in files:
        mode = infer_mode_from_path(fp)
        signed = mode == "signed"
        pset = fp.parent.name if fp.parent is not None else "unknown"

        instances = load_instances_from_txt(fp)
        out_dir = Path(args.out_root) / mode / pset / fp.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing: {fp}")

        for i, (H, G) in enumerate(instances, start=1):
            if args.only_instance > 0 and i != args.only_instance:
                continue

            out_path = out_dir / f"inst{i}.json"

            result = process_instance(
                H,
                G,
                signed=signed,
                minimal_only=args.minimal_only,
                pseudo_l=args.pseudo_l,
                pseudo_use_abs=args.pseudo_use_abs,
            )

            result["instance_id"] = i
            result["source_file"] = str(fp)
            result["len_A"] = len(H)
            result["len_B"] = len(G)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            print(
                f"  Instance {i}: verts={result['num_vertices']}, "
                f"edges={result['num_edges']}, "
                f"greedy_size={result['greedy_solution']['size']}, "
                f"pseudo_l={args.pseudo_l}, "
                f"valid={result['greedy_solution']['satisfies_pseudo_l']}"
            )


if __name__ == "__main__":
    main()