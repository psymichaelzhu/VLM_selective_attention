# %% Objective
"""
generate_design_matrix.py

Generate a trial design matrix for one experiment, then split it into
session files for parallel execution. After saving, writes session_ids
back into configs/experiments.yml.

No CLIP model is involved. Output is purely trial-level metadata.

CLI usage:
    python generate_design_matrix.py --experiment_id D1_M1_pairwise_config1
    python generate_design_matrix.py --experiment_id D1_M1_grid_config1
"""

# %% Imports
import sys
import argparse
import itertools
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import List

from helper.general import load_configs, load_item_ids


# %% Constants
GRID_SIZE = 9
GRID_MIN_ITEMS = 4
GRID_MAX_ITEMS = 9


# %% Pairwise design

def generate_pairwise_design(item_ids: List[str], seed: int) -> pd.DataFrame:
    """
    Generate all ordered item pairs (permutations), excluding self-pairs.

    Inputs:
        item_ids : list of item ID strings
        seed     : random seed for shuffling

    Outputs:
        DataFrame with columns [trial_id, item_1, item_2]
    """
    pairs = [(a, b) for a, b in itertools.permutations(item_ids, 2)]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in idx]

    df = pd.DataFrame(pairs, columns=["item_1", "item_2"])
    df.insert(0, "trial_id", range(len(df)))
    print(f"Generated {len(df)} pairwise trials")
    return df


# %% Grid design

def generate_grid_design(
    item_ids: List[str],
    n_trials: int,
    seed: int,
) -> pd.DataFrame:
    """
    Generate n_trials unique random grid trials (3x3, 4–9 items each).

    Two trials are identical if the same item occupies the same position in both.
    Deduplication uses tuple-based hashing.

    Inputs:
        item_ids : list of item ID strings
        n_trials : target number of unique trials
        seed     : random seed

    Outputs:
        DataFrame with columns [trial_id, item_1, ..., item_9]
    """
    if len(item_ids) < GRID_MIN_ITEMS:
        raise ValueError(f"Need at least {GRID_MIN_ITEMS} items for grid task, got {len(item_ids)}")

    rng = np.random.default_rng(seed)
    seen = set()
    rows = []

    chunk_size = max(n_trials * 2, 10_000)

    while len(rows) < n_trials:
        n_items_arr = rng.integers(GRID_MIN_ITEMS, GRID_MAX_ITEMS + 1, size=chunk_size)

        for n_items in n_items_arr:
            if len(rows) >= n_trials:
                break
            selected = rng.choice(item_ids, size=n_items, replace=False).tolist()
            positions = rng.choice(GRID_SIZE, size=n_items, replace=False) + 1  # 1-based

            row = {f"item_{i}": None for i in range(1, 10)}
            for item, pos in zip(selected, positions):
                row[f"item_{pos}"] = item

            key = tuple(row[f"item_{i}"] or "" for i in range(1, 10))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    df = pd.DataFrame(rows[:n_trials])
    df.insert(0, "trial_id", range(len(df)))
    print(f"Generated {len(df)} unique grid trials")
    return df


# %% Session splitting and saving

def split_and_save_sessions(
    design_df: pd.DataFrame,
    session_size: int,
    experiment_id: str,
) -> List[int]:
    """
    Split design matrix into fixed-size session CSVs and save to disk.

    Inputs:
        design_df     : full design matrix DataFrame
        session_size  : number of trials per session
        experiment_id : experiment key (determines output directory)

    Outputs:
        List of session_id integers that were saved
    """
    out_dir = Path("experiment") / experiment_id / "design_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)

    session_ids = []
    n = len(design_df)
    for session_id, start in enumerate(range(0, n, session_size)):
        chunk = design_df.iloc[start: start + session_size].copy()
        path = out_dir / f"{session_id}.csv"
        chunk.to_csv(path, index=False)
        print(f"  Saved session {session_id}: {len(chunk)} trials → {path}")
        session_ids.append(session_id)

    print(f"Split into {len(session_ids)} sessions of up to {session_size} trials each")
    return session_ids


def write_session_ids_to_yaml(experiment_id: str, session_ids: List[int]) -> None:
    """
    Write session_ids list back into configs/experiments.yml under the experiment key.

    Only the session_ids field is updated; all other fields are left untouched.

    Inputs:
        experiment_id : experiment key
        session_ids   : list of integer session IDs
    """
    yml_path = Path("configs") / "experiments.yml"
    with open(yml_path) as f:
        all_experiments = yaml.safe_load(f)

    all_experiments[experiment_id]["experiment"]["session_ids"] = session_ids

    with open(yml_path, "w") as f:
        yaml.dump(all_experiments, f, default_flow_style=False, sort_keys=False)

    print(f"Wrote session_ids {session_ids} to {yml_path}")


# %% CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate trial design matrix and split into session files."
    )
    parser.add_argument("--experiment_id", required=True,
                        help="Key in configs/experiments.yml")
    return parser.parse_args()


# %% Main

def main():
    args = parse_args()
    exp_cfg, _, dataset_cfg, setup_cfg = load_configs(args.experiment_id)

    task = exp_cfg["experiment"]["task"]
    dataset_name = dataset_cfg["name"]
    subset_name = dataset_cfg["subset"]
    seed = setup_cfg["run"]["seed"]
    session_size = setup_cfg["run"]["session_size"]
    n_trials = setup_cfg["run"]["n_trials"]

    item_ids = load_item_ids(dataset_name, subset_name)

    if task == "pairwise":
        design_df = generate_pairwise_design(item_ids, seed=seed)
    elif task == "grid":
        if n_trials is None:
            print("Error: n_trials must be set in setups.yml for grid task", file=sys.stderr)
            sys.exit(1)
        design_df = generate_grid_design(item_ids, n_trials=n_trials, seed=seed)
    else:
        print(f"Error: unknown task '{task}'", file=sys.stderr)
        sys.exit(1)

    session_ids = split_and_save_sessions(design_df, session_size, args.experiment_id)
    write_session_ids_to_yaml(args.experiment_id, session_ids)
    print("Done.")


if __name__ == "__main__":
    main()
