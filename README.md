# Heuristic Top-K Enumeration for the One-sided Exemplar Adjacency Number Problem

This repository contains the implementation and experimental scripts for the paper:

> *Heuristic Top-K Enumeration for the One-sided Exemplar Adjacency Number Problem*

The project studies heuristic Top-K maximal independent set (MIS) enumeration for the one-sided Exemplar Adjacency Number (EAN) problem under pseudo-\ell exemplar constraints.

---

# Features

- Conflict graph construction for one-sided EAN
- Bron--Kerbosch based heuristic MIS enumeration
- Pseudo-1 and pseudo-2 exemplar constraints
- Signed and unsigned genome variants
- Greedy MIS-CIG baseline implementation
- Signature-based folded exemplar deduplication
- Experimental comparison scripts

---

# Repository Structure

```text
enumerate_mis.py              # Heuristic Top-K MIS enumeration
qingge_greedy_pseudo.py     # Greedy MIS-CIG baseline
decode_mis.py                 # Decode MIS solutions into folded exemplars
make_pseudo_comparison_csv.py # Generate comparison CSV tables

qingge_datasets/              # Synthetic benchmark datasets
results/                      # Output JSON and comparison tables
paper/                        # LaTeX source of the paper
