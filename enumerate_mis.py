#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Iterable, Optional, Callable
import hashlib
import argparse


# ----------------------------
# Parsing genomes (signed/unsigned)
# ----------------------------

def parse_genome(s: str) -> List[int]:
    """
    Parse a genome string into a list of signed integers.
    Accepts:
      - "2143245"  -> [2,1,4,3,2,4,5]
      - "+2 -1 +3" -> [2,-1,3]
      - "-1+2-3"   -> [-1,2,-3]   (no spaces)
    """
    s = s.strip()
    if not s:
        return []

    # If there are spaces, use split tokens.
    if " " in s:
        toks = [t for t in s.replace(",", " ").split() if t]
        out = []
        for t in toks:
            out.append(int(t))
        return out

    # No spaces: could be "2143..." (unsigned digits) or "-1+2-3"
    # If it contains + or -, parse as signed tokens.
    if "+" in s or "-" in s:
        out = []
        i = 0
        n = len(s)
        while i < n:
            sign = 1
            if s[i] == '+':
                sign = 1
                i += 1
            elif s[i] == '-':
                sign = -1
                i += 1
            # read number
            if i >= n or not s[i].isdigit():
                raise ValueError(f"Bad signed genome encoding near index {i} in: {s}")
            j = i
            while j < n and s[j].isdigit():
                j += 1
            out.append(sign * int(s[i:j]))
            i = j
        return out

    # Otherwise, treat as unsigned digits: "2143245" -> [2,1,4,3,2,4,5]
    if not s.isdigit():
        raise ValueError(f"Cannot parse genome: {s}")
    return [int(ch) for ch in s]


# ----------------------------
# Adjacency definitions
# ----------------------------

def canonical_signed_adj(a: int, b: int) -> Tuple[int, int]:
    """
    Signed adjacency equivalence: (a,b) ~ (-b,-a).
    Return a canonical representation of the undirected signed adjacency.
    """
    x1 = (a, b)
    x2 = (-b, -a)
    return x1 if x1 <= x2 else x2


def canonical_unsigned_adj(a: int, b: int) -> Tuple[int, int]:
    """Unsigned adjacency is direction-free: {a,b}."""
    return (a, b) if a <= b else (b, a)


# ----------------------------
# Vertex & conflict graph
# ----------------------------

@dataclass(frozen=True)
class Vertex:
    vid: int
    color: int      # index of adjacency in A: 0..len(A)-2
    L: int          # endpoint index in B (0-based, inclusive)
    R: int          # endpoint index in B (0-based, inclusive)
    adj_key: Tuple[int, int]  # canonical adjacency representation

    def interior_open(self) -> Tuple[int, int]:
        """Open interior (L,R) used for overlap conflicts."""
        return (self.L, self.R)


def open_interiors_overlap(u: Vertex, v: Vertex) -> bool:
    """
    Open interval overlap for interiors (L,R) with endpoints excluded.
    Touching at endpoints is allowed (so chaining is possible).
    For open intervals: (Lu,Ru) intersects (Lv,Rv) iff max(Lu,Lv) < min(Ru,Rv).
    """
    return max(u.L, v.L) < min(u.R, v.R)


def build_conflict_graph(vertices: List[Vertex], enforce_same_color_as_conflict: bool = True) -> Dict[int, Set[int]]:
    """
    Conflict graph: edge if open interiors overlap (cannot simultaneously delete consistently),
    plus optional edge if same color (at most one realization per reference adjacency).
    """
    adj: Dict[int, Set[int]] = {v.vid: set() for v in vertices}
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            a, b = vertices[i], vertices[j]
            conflict = open_interiors_overlap(a, b)
            if enforce_same_color_as_conflict and a.color == b.color:
                conflict = True
            if conflict:
                adj[a.vid].add(b.vid)
                adj[b.vid].add(a.vid)
    return adj


# ----------------------------
# Candidate interval generation
# ----------------------------

def build_pos_index(B: List[int], signed: bool) -> Dict[int, List[int]]:
    """
    Map gene key -> positions in B.
    For signed=True, we index by signed value (e.g., -3 and +3 are different).
    For signed=False, we index by absolute value (gene family).
    """
    pos: Dict[int, List[int]] = {}
    for i, x in enumerate(B):
        key = x if signed else abs(x)
        pos.setdefault(key, []).append(i)
    return pos


def generate_vertices(
    A: List[int],
    B: List[int],
    signed: bool,
    all_pairs: bool = True,
    minimal_only: bool = False,
) -> List[Vertex]:
    """
    Generate candidate vertices (intervals in B) for each adjacency in A.

    - signed=False: adjacency color is {abs(A[i]), abs(A[i+1])}
    - signed=True : adjacency color uses canonical_signed_adj(A[i], A[i+1])

    all_pairs=True:
      include all endpoint pairs (i<j) in B that match the adjacency key.
    minimal_only=True:
      keep only locally minimal candidates per occurrence (reduces vertices).
      For enumeration correctness, prefer minimal_only=False unless you know reduction guarantees it.
    """
    if len(A) < 2:
        return []

    pos = build_pos_index(B, signed=signed)
    vertices: List[Vertex] = []
    vid = 0

    for c in range(len(A) - 1):
        a1, a2 = A[c], A[c + 1]
        if signed:
            # signed adjacency defined on signed endpoints
            adj_key = canonical_signed_adj(a1, a2)
            # candidates in B must match either (a1,a2) or (-a2,-a1) up to positions
            # We'll match using canonical_signed_adj(B[i],B[j]) == adj_key
            # But we want efficiency: limit endpoint candidates by possible signed values involved.
            # Adj endpoints can be either (a1,a2) or (-a2,-a1).
            endpoint_options = {(a1, a2), (-a2, -a1)}
            # Index positions by exact signed values
            # Candidate endpoints must be one of the two options (in either order along B, because we allow adjacency direction-free via canonical)
            needed_vals = set([a1, a2, -a1, -a2])  # for robust indexing
            # We'll enumerate via abs? No; signed model treats +g and -g separately
        else:
            adj_key = canonical_unsigned_adj(abs(a1), abs(a2))
            endpoint_options = {(abs(a1), abs(a2)), (abs(a2), abs(a1))}
            needed_vals = {abs(a1), abs(a2)}

        # Collect endpoint positions for each needed endpoint value
        # For unsigned, key is abs(g). For signed, key is signed g itself.
        positions_by_val: Dict[int, List[int]] = {}
        for val in needed_vals:
            positions_by_val[val] = pos.get(val if signed else abs(val), [])

        # Enumerate candidates
        # For correctness (enumeration), you generally want all pairs where canonical adjacency matches.
        # For speed, minimal_only chooses nearest matches only.
        if not signed:
            u, v = abs(a1), abs(a2)
            Pu = positions_by_val.get(u, [])
            Pv = positions_by_val.get(v, [])

            if minimal_only:
                # For each occurrence of u, take nearest v to the right and left.
                for i in Pu:
                    # nearest to right
                    right = [j for j in Pv if j > i]
                    if right:
                        j = right[0]
                        L, R = (i, j) if i < j else (j, i)
                        vertices.append(Vertex(vid, c, L, R, adj_key))
                        vid += 1
                    # nearest to left
                    left = [j for j in Pv if j < i]
                    if left:
                        j = left[-1]
                        L, R = (i, j) if i < j else (j, i)
                        vertices.append(Vertex(vid, c, L, R, adj_key))
                        vid += 1
            else:
                # all pairs (i,j) across Pu x Pv
                for i in Pu:
                    for j in Pv:
                        if i == j:
                            continue
                        L, R = (i, j) if i < j else (j, i)
                        vertices.append(Vertex(vid, c, L, R, adj_key))
                        vid += 1
        else:
            # Signed: candidates must satisfy canonical_signed_adj(B[i], B[j]) == adj_key
            # We'll enumerate positions for the specific signed endpoint values that appear in adj_key.
            # adj_key is a canonical pair (p,q).
            p, q = adj_key
            Pp = pos.get(p, [])
            Pq = pos.get(q, [])

            # But remember canonical means also (-q,-p) equivalent; those are already captured by canonical check.
            # We can simply enumerate all i in positions of any signed gene and test canonical.
            if minimal_only:
                # nearest matches only: for each occurrence of p, nearest q to right/left
                for i in Pp:
                    right = [j for j in Pq if j > i]
                    if right:
                        j = right[0]
                        L, R = (i, j) if i < j else (j, i)
                        vertices.append(Vertex(vid, c, L, R, adj_key))
                        vid += 1
                    left = [j for j in Pq if j < i]
                    if left:
                        j = left[-1]
                        L, R = (i, j) if i < j else (j, i)
                        vertices.append(Vertex(vid, c, L, R, adj_key))
                        vid += 1
            else:
                # all pairs across Pp x Pq
                for i in Pp:
                    for j in Pq:
                        if i == j:
                            continue
                        L, R = (i, j) if i < j else (j, i)
                        # sanity check on canonical
                        if canonical_signed_adj(B[L], B[R]) == adj_key:
                            vertices.append(Vertex(vid, c, L, R, adj_key))
                            vid += 1

    # Optional: remove duplicate vertices (same color, same endpoints)
    uniq = {}
    for v in vertices:
        key = (v.color, v.L, v.R, v.adj_key)
        uniq[key] = v
    vertices = list(uniq.values())
    # Re-assign ids for cleanliness
    vertices = [Vertex(i, v.color, v.L, v.R, v.adj_key) for i, v in enumerate(sorted(vertices, key=lambda x: (x.color, x.L, x.R)))]
    return vertices


# ----------------------------
# MIS enumeration (Tsukiyama/Lawler style via modern recursion)
# ----------------------------

def choose_pivot(adj: Dict[int, Set[int]], C: Set[int], X: Set[int]) -> Optional[int]:
    """
    Pivot heuristic: choose u in C ∪ X maximizing |C ∩ N(u)|.
    """
    U = C | X
    if not U:
        return None
    best_u = None
    best_score = -1
    for u in U:
        score = len(C & adj[u])
        if score > best_score:
            best_score = score
            best_u = u
    return best_u


def non_neighbors(adj: Dict[int, Set[int]], v: int, S: Set[int]) -> Set[int]:
    """
    Return subset of S that are non-neighbors of v (and not v itself).
    """
    Nv = adj[v]
    return {u for u in S if u != v and u not in Nv}


def enumerate_maximal_independent_sets(
    adj: Dict[int, Set[int]],
    vertices: Set[int],
    limit_mis: Optional[int] = None,
) -> Iterable[Set[int]]:
    """
    Enumerate all maximal independent sets of the conflict graph.
    Uses a Bron–Kerbosch-style recursion on the complement graph WITHOUT building it.
    """
    out = 0

    def rec(I: Set[int], C: Set[int], X: Set[int]):
        nonlocal out
        if limit_mis is not None and out >= limit_mis:
            return

        if not C and not X:
            out += 1
            yield set(I)
            return

        u = choose_pivot(adj, C, X)
        # Branch vertices in C that are NON-neighbors of pivot u
        branch = list(C) if u is None else list(C - adj[u])

        for v in branch:
            C2 = non_neighbors(adj, v, C)
            X2 = non_neighbors(adj, v, X)
            yield from rec(I | {v}, C2, X2)

            # Move v from C to X (prevents duplicates)
            C.remove(v)
            X.add(v)

            if limit_mis is not None and out >= limit_mis:
                return

    yield from rec(set(), set(vertices), set())


# ----------------------------
# Convert MIS -> sequence by applying deletions
# ----------------------------

def apply_interval_deletions(B: List[int], mis: Set[int], vertices_by_id: Dict[int, Vertex]) -> List[int]:
    """
    Given MIS (set of chosen vertices), delete the interiors of each chosen interval:
      for each vertex with endpoints (L,R), delete indices (L+1 .. R-1).
    Then return the remaining sequence.
    """
    n = len(B)
    keep = [True] * n
    for vid in mis:
        v = vertices_by_id[vid]
        if v.R - v.L >= 2:
            for k in range(v.L + 1, v.R):
                keep[k] = False
    return [B[i] for i in range(n) if keep[i]]


def seq_signature(seq: List[int]) -> str:
    """
    Dedup signature. Use sha256 on comma-joined integers.
    """
    b = ",".join(map(str, seq)).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


# ----------------------------
# Full pipeline
# ----------------------------

def enumerate_scaffolds_via_maximal_is(
    A: List[int],
    B: List[int],
    signed: bool,
    limit_mis: Optional[int] = None,
    limit_unique: Optional[int] = None,
    minimal_only: bool = False,
) -> List[List[int]]:
    """
    Full pipeline:
      1) Generate candidate vertices
      2) Build conflict graph
      3) Enumerate maximal independent sets
      4) Map to sequences by deletions
      5) Deduplicate sequences

    Returns unique sequences (order = first seen).
    """
    vertices = generate_vertices(A, B, signed=signed, minimal_only=minimal_only)
    if not vertices:
        return []

    adj = build_conflict_graph(vertices, enforce_same_color_as_conflict=True)
    vertices_by_id = {v.vid: v for v in vertices}
    all_vids = set(vertices_by_id.keys())

    seen: Dict[str, List[int]] = {}
    for mis in enumerate_maximal_independent_sets(adj, all_vids, limit_mis=limit_mis):
        seq = apply_interval_deletions(B, mis, vertices_by_id)
        sig = seq_signature(seq)
        if sig not in seen:
            seen[sig] = seq
            if limit_unique is not None and len(seen) >= limit_unique:
                break

    return list(seen.values())


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Enumerate pseudo-exemplar scaffolds via maximal independent sets (MIS).")
    ap.add_argument("--A", required=True, help='Reference exemplar genome. Examples: "12345" or "+1 +2 -3 +4".')
    ap.add_argument("--B", required=True, help='Duplicated genome. Examples: "2143245" or "-1+2-3+4".')
    ap.add_argument("--signed", action="store_true", help="Use signed gene model (orientation-aware).")
    ap.add_argument("--limit_mis", type=int, default=None, help="Stop after enumerating this many MIS.")
    ap.add_argument("--limit_unique", type=int, default=50, help="Stop after this many unique sequences.")
    ap.add_argument("--minimal_only", action="store_true", help="Generate only nearest (locally minimal) candidates per adjacency (faster, may miss solutions).")
    args = ap.parse_args()

    A = parse_genome(args.A)
    B = parse_genome(args.B)

    seqs = enumerate_scaffolds_via_maximal_is(
        A=A,
        B=B,
        signed=args.signed,
        limit_mis=args.limit_mis,
        limit_unique=args.limit_unique,
        minimal_only=args.minimal_only,
    )

    print(f"A = {A}")
    print(f"B = {B}")
    print(f"signed = {args.signed}")
    print(f"unique scaffolds returned = {len(seqs)}")
    print("-" * 60)
    for i, s in enumerate(seqs, 1):
        print(f"{i:3d}: {s}")


if __name__ == "__main__":
    main()
