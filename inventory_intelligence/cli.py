from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from .service import generate_recommendations

def main():
    parser = argparse.ArgumentParser(description="AI Inventory Intelligence recommendation runner")
    parser.add_argument("--sales", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--products", required=True)
    parser.add_argument("--suppliers", required=True)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--service-level", type=float, default=0.95)
    parser.add_argument("--minimum-history-days", type=int, default=45)
    parser.add_argument("--output", default="outputs/recommendations.json")
    parser.add_argument("--csv", default=None, help="Optional CSV export path")
    args = parser.parse_args()
    result = generate_recommendations(
        args.sales, args.inventory, args.products, args.suppliers,
        horizon=args.horizon, service_level=args.service_level,
        minimum_history_days=args.minimum_history_days
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.json_normalize(result["inventory_recommendations"]).to_csv(csv_path, index=False)
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote recommendations to {out}")
    if args.csv:
        print(f"Wrote CSV to {args.csv}")

if __name__ == "__main__":
    main()
