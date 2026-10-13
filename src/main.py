import argparse
import os
from utils.quick_start import quick_start

os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["NUMEXPR_MAX_THREADS"] = "48"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="ToCoFMR", help="model name")
    parser.add_argument("--dataset", "-d", default="baby", help="dataset name")
    args = parser.parse_args()

    quick_start(model=args.model, dataset=args.dataset)
