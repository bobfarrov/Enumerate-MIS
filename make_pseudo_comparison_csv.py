import re
import csv
import sys
from pathlib import Path


def parse_greedy(file_path):
    results = []
    current_file = None
    current_pseudo_l = None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            m_pseudo = re.search(r"\[INFO\]\s+pseudo_l=(\d+)", line)
            if m_pseudo:
                current_pseudo_l = int(m_pseudo.group(1))

            m_file = re.search(r"Processing:\s+(.+)", line)
            if m_file:
                current_file = Path(m_file.group(1)).name

            m = re.search(
                r"Instance\s+(\d+):\s+verts=(\d+),\s+edges=(\d+),\s+greedy_size=(\d+)",
                line,
            )
            if m:
                results.append({
                    "dataset": current_file,
                    "instance": int(m.group(1)),
                    "vertices": int(m.group(2)),
                    "edges": int(m.group(3)),
                    "greedy": int(m.group(4)),
                    "pseudo_l": current_pseudo_l,
                })

    return results


def parse_enum(file_path):
    results = []
    current_file = None
    current_instance = None
    current_vertices = None
    current_pseudo_l = None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            m_pseudo = re.search(r"\[INFO\]\s+pseudo_l=(\d+)", line)
            if m_pseudo:
                current_pseudo_l = int(m_pseudo.group(1))

            m_file = re.search(r"Processing:\s+(.+)", line)
            if m_file:
                current_file = Path(m_file.group(1)).name

            m_inst = re.search(r"Instance\s+(\d+)\s+\(\|A\|=(\d+)\s+\|B\|=(\d+)\)", line)
            if m_inst:
                current_instance = int(m_inst.group(1))

            m_verts = re.search(r"verts=(\d+)", line)
            if m_verts:
                current_vertices = int(m_verts.group(1))

            # New format:
            # Enumerated=30, Accepted=30, Rejects=0, Top MIS size=575, K=30
            m_res = re.search(
                r"Enumerated=(\d+),\s+Accepted=(\d+),\s+Rejects=(\d+),\s+Top MIS size=(\d+),\s+K=(\d+)",
                line,
            )

            if m_res:
                results.append({
                    "dataset": current_file,
                    "instance": current_instance,
                    "vertices_enum": current_vertices,
                    "enumerated": int(m_res.group(1)),
                    "accepted": int(m_res.group(2)),
                    "rejects": int(m_res.group(3)),
                    "ours": int(m_res.group(4)),
                    "K": int(m_res.group(5)),
                    "pseudo_l": current_pseudo_l,
                })
                continue

            # Old format fallback:
            # Enumerated=30, Top MIS size=575, K=30
            m_old = re.search(
                r"Enumerated=(\d+),\s+Top MIS size=(\d+),\s+K=(\d+)",
                line,
            )

            if m_old:
                results.append({
                    "dataset": current_file,
                    "instance": current_instance,
                    "vertices_enum": current_vertices,
                    "enumerated": int(m_old.group(1)),
                    "accepted": "",
                    "rejects": "",
                    "ours": int(m_old.group(2)),
                    "K": int(m_old.group(3)),
                    "pseudo_l": current_pseudo_l,
                })

    return results


def main():
    if len(sys.argv) != 5:
        print("Usage:")
        print("python make_pseudo_comparison_csv.py greedy.txt enum.txt label output.csv")
        print()
        print("Example:")
        print("python make_pseudo_comparison_csv.py greedy_signed_pseudo1.txt enum_signed_pseudo1_k30.txt signed_pseudo1 signed_pseudo1.csv")
        sys.exit(1)

    greedy_file = sys.argv[1]
    enum_file = sys.argv[2]
    label = sys.argv[3]
    output_file = sys.argv[4]

    greedy = parse_greedy(greedy_file)
    enum = parse_enum(enum_file)

    enum_map = {
        (x["dataset"], x["instance"]): x
        for x in enum
    }

    rows = []

    for g in greedy:
        key = (g["dataset"], g["instance"])
        e = enum_map.get(key)

        if e is None:
            continue

        improvement = e["ours"] - g["greedy"]
        percent = (improvement / g["greedy"]) * 100 if g["greedy"] else 0

        rows.append({
            "experiment": label,
            "dataset": g["dataset"],
            "instance": g["instance"],
            "pseudo_l": e["pseudo_l"] if e["pseudo_l"] is not None else g["pseudo_l"],
            "vertices": g["vertices"],
            "edges": g["edges"],
            "greedy": g["greedy"],
            "ours": e["ours"],
            "improvement": improvement,
            "percent_improvement": round(percent, 2),
            "enumerated": e["enumerated"],
            "accepted": e["accepted"],
            "rejects": e["rejects"],
            "K": e["K"],
        })

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "experiment",
            "dataset",
            "instance",
            "pseudo_l",
            "vertices",
            "edges",
            "greedy",
            "ours",
            "improvement",
            "percent_improvement",
            "enumerated",
            "accepted",
            "rejects",
            "K",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved comparison table to {output_file}")
    print(f"Matched rows: {len(rows)}")

    if rows:
        wins = sum(1 for r in rows if r["improvement"] > 0)
        ties = sum(1 for r in rows if r["improvement"] == 0)
        losses = sum(1 for r in rows if r["improvement"] < 0)
        avg_gain = sum(r["improvement"] for r in rows) / len(rows)
        avg_pct = sum(r["percent_improvement"] for r in rows) / len(rows)

        print(f"Wins: {wins}, Ties: {ties}, Losses: {losses}")
        print(f"Average improvement: {avg_gain:.2f}")
        print(f"Average percent improvement: {avg_pct:.2f}%")


if __name__ == "__main__":
    main()