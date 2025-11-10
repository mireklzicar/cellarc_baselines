import os
from pathlib import Path

import pandas as pd
import wandb


ENTITY = "lzicar"
PROJECTS = [
    "cellarc100k_50e_embedding_small",
    "cellarc100k_50e_incontext_medium",
    "cellarc100k_50e_embedding_large",
    "cellarc100k_50e_incontext_small",
    "cellarc100k_50e_incontext_large",
    "cellarc100k_50e_embedding_medium",
]


def collect_project_runs(api: wandb.Api, entity: str, project: str) -> pd.DataFrame:
    """Fetch all runs for a single project and flatten config/summary."""
    runs = api.runs(f"{entity}/{project}")
    rows = []
    for r in runs:
        row = {
            "run_id": r.id,
            "name": r.name,
            "state": r.state,
            "created_at": r.created_at,
            "tags": ",".join(r.tags or []),
            "user": getattr(r, "user", None).username if getattr(r, "user", None) else None,
            "url": r.url,
        }
        cfg = r.config or {}
        summ = r.summary or {}
        row.update({f"config.{k}": v for k, v in cfg.items()})
        row.update({f"summary.{k}": v for k, v in summ.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame, out_base: Path) -> None:
    """Write the dataframe to CSV and JSON with a shared base filename."""
    df.to_csv(out_base.with_suffix(".csv"), index=False)
    df.to_json(out_base.with_suffix(".json"), orient="records", indent=2)


def main() -> None:
    api = wandb.Api()
    out_dir = Path("wandb_export")
    os.makedirs(out_dir, exist_ok=True)

    for project in PROJECTS:
        df = collect_project_runs(api, ENTITY, project)
        write_outputs(df, out_dir / project)
        print(f"Wrote {project}.csv and {project}.json to {out_dir}/")


if __name__ == "__main__":
    main()
