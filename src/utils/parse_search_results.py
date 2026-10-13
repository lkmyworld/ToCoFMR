import argparse
import ast
import csv
import glob
import os
import re


PARAM_RE = re.compile(r"Parameters: (?P<names>\[.*?\]) = (?P<values>\(.*?\))")
METRIC_RE = re.compile(r"(?P<split>Best valid|Best test|Valid|Test)\s+(?P<metrics>[A-Za-z0-9_]+: \[.*)")
METRIC_VALUES_RE = re.compile(r"([A-Za-z0-9_]+): \[([^\]]*)\]")
TOPK_RE = re.compile(r"^\s*topk:\s*(\[.*?\])\s*$")
DEFAULT_TOPK = [1, 2, 5, 10, 20, 50, 100]


def add_metrics(row, prefix, text, topk):
    for metric, values in METRIC_VALUES_RE.findall(text):
        values = [float(value.strip()) for value in values.split(',')]
        if len(values) != len(topk):
            raise ValueError(f"{metric}: {len(values)} values for {len(topk)} Top-K entries")
        for k, value in zip(topk, values):
            row[f"{prefix}_{metric.lower()}@{k}"] = value


def has_test_metrics(row):
    return row and any(key.startswith(('test_', 'best_test_')) for key in row)


def parse_log(path):
    rows = []
    current = None
    topk = DEFAULT_TOPK
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            topk_match = TOPK_RE.search(line)
            if topk_match:
                topk = ast.literal_eval(topk_match.group(1))
                continue
            param_match = PARAM_RE.search(line)
            if param_match:
                if has_test_metrics(current):
                    rows.append(current)
                current = {
                    "log_file": os.path.basename(path),
                    "param_names": ", ".join(ast.literal_eval(param_match.group("names"))),
                    "param_values": repr(ast.literal_eval(param_match.group("values"))),
                }
                continue

            metric_match = METRIC_RE.search(line)
            if current and metric_match:
                split = metric_match.group("split").strip().lower().replace(" ", "_")
                add_metrics(current, split, metric_match.group("metrics"), topk)

    if has_test_metrics(current):
        rows.append(current)
    return rows


def write_csv(rows, output_path):
    preferred = [
        "log_file",
        "param_names",
        "param_values",
        "valid_recall@20",
        "valid_ndcg@20",
        "test_recall@20",
        "test_ndcg@20",
        "best_valid_recall@20",
        "best_valid_ndcg@20",
        "best_test_recall@20",
        "best_test_ndcg@20",
    ]
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    fieldnames = [key for key in preferred if key in keys] + [key for key in keys if key not in preferred]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="log")
    parser.add_argument("--pattern", default="tocofmr-*.log")
    parser.add_argument("--output", default="search_results/tocofmr_search_results.csv")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.log_dir, args.pattern))):
        rows.extend(parse_log(path))

    write_csv(rows, args.output)
    print(f"parsed {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
