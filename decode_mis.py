#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict


# -----------------------
# Helpers: gene naming
# -----------------------
def gene_name(gid: int, width: int = 4) -> str:
    return f"g{gid:0{width}d}"


def format_gene_token(x: int, width: int = 4) -> str:
    name = gene_name(abs(x), width)
    return f"-{name}" if x < 0 else name


# -----------------------
# Parsing
# -----------------------
def parse_genome(s: str) -> List[int]:
    tokens = re.findall(r"[+-]?\d+", s)
    return [int(t) for t in tokens]


# -----------------------
# Vertex model
# -----------------------
@dataclass(frozen=True)
class Vertex:
    vid: int
    color: int     # index c in H for adjacency (H[c], H[c+1])
    L: int         # left endpoint index in G
    R: int         # right endpoint index in G
    adj_key: Tuple[int, int]


def canonical_signed_adj(a: int, b: int) -> Tuple[int, int]:
    x1, x2 = (a, b), (-b, -a)
    return x1 if x1 <= x2 else x2


def generate_vertices(H: List[int], G: List[int], signed: bool, minimal_only: bool) -> List[Vertex]:
    pos = defaultdict(list)
    for i, x in enumerate(G):
        k = x if signed else abs(x)
        pos[k].append(i)

    vertices: List[Vertex] = []
    vid = 0

    for c in range(len(H) - 1):
        h1, h2 = H[c], H[c + 1]

        if signed:
            adj_key = canonical_signed_adj(h1, h2)
            pairs = {(adj_key[0], adj_key[1]), (-adj_key[1], -adj_key[0])}
        else:
            adj_key = tuple(sorted((abs(h1), abs(h2))))
            pairs = {(abs(h1), abs(h2)), (abs(h2), abs(h1))}

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

    # de-dup exact duplicates
    unique_v: Dict[Tuple[int, int, int, Tuple[int, int]], Vertex] = {}
    for v in vertices:
        unique_v[(v.color, v.L, v.R, v.adj_key)] = v

    out = [
        Vertex(i, v.color, v.L, v.R, v.adj_key)
        for i, v in enumerate(sorted(unique_v.values(), key=lambda x: (x.color, x.L, x.R)))
    ]
    return out


def read_instance(dataset_file: str, instance_id: int) -> Tuple[List[int], List[int]]:
    instances = []
    with open(dataset_file, "r", encoding="utf-8-sig") as f:
        H_curr = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("H:"):
                H_curr = parse_genome(line)
            elif line.upper().startswith("G:"):
                if H_curr is None:
                    continue
                G_curr = parse_genome(line)
                instances.append((H_curr, G_curr))
                H_curr = None

    if not (1 <= instance_id <= len(instances)):
        raise ValueError(f"instance_id={instance_id} out of range (1..{len(instances)})")

    return instances[instance_id - 1]


def fold_exemplar_from_H(H: List[int], chosen_vertices: List[Vertex], *, unsigned_fold_abs: bool) -> List[int]:
    keep = set()
    for v in chosen_vertices:
        c = v.color
        keep.add(H[c])
        keep.add(H[c + 1])

    if unsigned_fold_abs:
        keep_abs = {abs(x) for x in keep}
        return [x for x in H if abs(x) in keep_abs]
    else:
        return [x for x in H if x in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_file", required=True)
    ap.add_argument("--result_json", required=True)
    ap.add_argument("--instance_id", type=int, default=1)
    ap.add_argument("--which", type=int, default=0, help="which top_mis index to decode (0=best)")
    ap.add_argument("--unsigned_fold_abs", action="store_true",
                    help="Use abs-folding when UNSIGNED (leave OFF for signed).")
    ap.add_argument("--gene_width", type=int, default=4)
    ap.add_argument("--no_decode_table", action="store_true",
                    help="If set, do NOT print the long per-vertex decode table.")
    ap.add_argument("--signed", action="store_true", help="fallback only if JSON missing 'signed'")
    ap.add_argument("--minimal_only", action="store_true", help="fallback only if JSON missing 'minimal_only'")
    args = ap.parse_args()

    with open(args.result_json, "r", encoding="utf-8") as f:
        res = json.load(f)

    signed = bool(res.get("signed", args.signed))
    minimal_only = bool(res.get("minimal_only", args.minimal_only))

    # Safety: warn if user accidentally uses unsigned_fold_abs in signed mode
    if signed and args.unsigned_fold_abs:
        print("[NOTE] signed=True but --unsigned_fold_abs was set. Usually you want this OFF for signed runs.\n")

    H, G = read_instance(args.dataset_file, args.instance_id)
    verts = generate_vertices(H, G, signed=signed, minimal_only=minimal_only)
    vmap = {v.vid: v for v in verts}

    top_mis = res.get("top_mis", [])
    if not top_mis:
        print("No top_mis in JSON (maybe skipped or Enumerated=0).")
        return

    if not (0 <= args.which < len(top_mis)):
        raise ValueError(f"--which={args.which} out of range (0..{len(top_mis)-1})")

    mis_list = top_mis[args.which]
    chosen = [vmap[vid] for vid in mis_list if vid in vmap]
    missing = [vid for vid in mis_list if vid not in vmap]

    print(f"Decoded MIS #{args.which}: size={len(chosen)}")
    print(f"|H|={len(H)}, |G|={len(G)}, generated_verts={len(verts)}")
    print(f"JSON says: signed={signed}, minimal_only={minimal_only}")
    if missing:
        print(f"[WARNING] {len(missing)} vids from JSON not found in regenerated vertices.")
        print("          This usually means wrong dataset_file or wrong instance_id.")
        print("          First missing vids:", missing[:10])

    chosen_sorted = sorted(chosen, key=lambda v: (v.color, v.L, v.R))

    if not args.no_decode_table:
        print("\nPer-vertex decode:")
        for v in chosen_sorted:
            c = v.color
            h1, h2 = H[c], H[c + 1]
            gL, gR = G[v.L], G[v.R]
            print(
                f"  vid={v.vid:5d}  c={c:5d}  H[c,c+1]=({h1},{h2})  "
                f"interval=[{v.L},{v.R}]  G[L],G[R]=({gL},{gR})  adj_key={v.adj_key}"
            )

    exemplar = fold_exemplar_from_H(H, chosen_sorted, unsigned_fold_abs=args.unsigned_fold_abs)
    named_exemplar = [format_gene_token(x, width=args.gene_width) for x in exemplar]

    print("\nFolded exemplar (numeric, subsequence of H):")
    print("  " + " ".join(map(str, exemplar)))

    print("\nFolded exemplar (named genes, subsequence of H):")
    print("  " + " ".join(named_exemplar))


if __name__ == "__main__":
    main()
