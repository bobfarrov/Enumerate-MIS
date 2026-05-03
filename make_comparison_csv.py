import re
import csv
import sys
from pathlib import Path

def parse_greedy(file_path):
    results = []
    current_file = None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

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
                })

    return results


def parse_enum(file_path):
    results = []
    current_file = None
    current_instance = None
    current_vertices = None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            m_file = re.search(r"Processing:\s+(.+)", line)
            if m_file:
                current_file = Path(m_file.group(1)).name

            m_inst = re.search(r"Instance\s+(\d+)\s+\(\|A\|=(\d+)\s+\|B\|=(\d+)\)", line)
            if m_inst:
                current_instance = int(m_inst.group(1))

            m_verts = re.search(r"verts=(\d+)", line)
            if m_verts:
                current_vertices = int(m_verts.group(1))

            m_res = re.search(r"Enumerated=(\d+),\s+Top MIS size=(\d+),\s+K=(\d+)", line)
            if m_res:
                results.append({
                    "dataset": current_file,
                    "instance": current_instance,
                    "vertices_enum": current_vertices,
                    "enumerated": int(m_res.group(1)),
                    "ours": int(m_res.group(2)),
                    "K": int(m_res.group(3)),
                })

    return results


def main():
    if len(sys.argv) != 4:
        print("Usage: python make_comparison_csv.py greedy.txt enum.txt output.csv")
        sys.exit(1)

    greedy_file = sys.argv[1]
    enum_file = sys.argv[2]
    output_file = sys.argv[3]

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
            "dataset": g["dataset"],
            "instance": g["instance"],
            "vertices": g["vertices"],
            "edges": g["edges"],
            "greedy": g["greedy"],
            "ours": e["ours"],
            "improvement": improvement,
            "percent_improvement": round(percent, 2),
            "enumerated": e["enumerated"],
            "K": e["K"],
        })

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset",
            "instance",
            "vertices",
            "edges",
            "greedy",
            "ours",
            "improvement",
            "percent_improvement",
            "enumerated",
            "K",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved comparison table to {output_file}")
    print(f"Matched rows: {len(rows)}")


if __name__ == "__main__":
    main()