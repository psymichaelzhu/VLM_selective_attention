# %% Objective
"""
helper/general.py

Shared utility functions used across generate_design_matrix.py,
prepare_embedding_cache.py, and run_session.py.

Functions:
    load_configs        — load experiment, model, dataset, and setup configs from YAML
    load_item_ids       — return sorted item ID list from subset.yml
    load_item_images    — load item images as uint8 arrays via subset.yml
    handle_device       — resolve 'auto' device to 'cpu' or 'cuda'
    normalize_rows      — L2-normalize rows of a 2D float array
"""

# %% Imports
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image


# %% Config loading

def load_configs(experiment_id: str) -> Tuple[dict, dict, dict, dict]:
    """
    Load experiment, model, dataset, and setup configs from YAML files.

    Inputs:
        experiment_id : key in configs/experiments.yml

    Outputs:
        (exp_cfg, model_cfg, dataset_cfg, setup_cfg)

        - exp_cfg    : experiment binding (dataset_id, model_id, setup_id, task, session_ids)
        - model_cfg  : model definition (model_spec, pretrained)
        - dataset_cfg: dataset definition (name, subset)
        - setup_cfg  : reusable setup (run, layout, metrics)

    Note: do not merge setup_cfg back into exp_cfg.
    Downstream code should read setup_cfg["run"], setup_cfg["layout"],
    and setup_cfg["metrics"] directly.
    """
    cfg_dir = Path("configs")

    # --- experiments.yml ---
    with open(cfg_dir / "experiments.yml",encoding="utf-8") as f:
        all_experiments = yaml.safe_load(f)
    if experiment_id not in all_experiments:
        raise KeyError(f"experiment_id '{experiment_id}' not found in experiments.yml")
    exp_cfg = all_experiments[experiment_id]

    # --- models.yml ---
    model_id = exp_cfg["experiment"]["model_id"]
    with open(cfg_dir / "models.yml",encoding="utf-8") as f:
        all_models = yaml.safe_load(f)
    if model_id not in all_models:
        raise KeyError(f"model_id '{model_id}' not found in models.yml")
    model_cfg = all_models[model_id]

    # --- dataset.yml ---
    dataset_id = exp_cfg["experiment"]["dataset_id"]
    with open(cfg_dir / "dataset.yml",encoding="utf-8") as f:
        all_datasets = yaml.safe_load(f)
    if dataset_id not in all_datasets:
        raise KeyError(f"dataset_id '{dataset_id}' not found in dataset.yml")
    dataset_cfg = all_datasets[dataset_id]

    # --- setups.yml ---
    setup_id = exp_cfg["experiment"]["setup_id"]
    with open(cfg_dir / "setups.yml",encoding="utf-8") as f:
        all_setups = yaml.safe_load(f)
    if setup_id not in all_setups:
        raise KeyError(f"setup_id '{setup_id}' not found in setups.yml")
    setup_cfg = all_setups[setup_id]

    return exp_cfg, model_cfg, dataset_cfg, setup_cfg


# %% Item loading

def load_item_ids(dataset_name: str, subset_name: Optional[str]) -> List[str]:
    """
    Return sorted list of item ID strings from subset.yml.

    Inputs:
        dataset_name : dataset name (maps to data/{dataset_name}/)
        subset_name  : key in subset.yml; if None, uses 'all'

    Outputs:
        Sorted list of item ID strings (filenames stripped of extension)
    """
    subset_path = Path("data") / dataset_name / "subset.yml"
    if not subset_path.exists():
        raise FileNotFoundError(f"subset.yml not found: {subset_path}")

    with open(subset_path,encoding="utf-8") as f:
        subsets = yaml.safe_load(f)

    key = subset_name if subset_name is not None else "all"
    if key not in subsets:
        raise KeyError(f"Subset '{key}' not found in {subset_path}")

    filenames = subsets[key]
    item_ids = sorted(Path(fn).stem for fn in filenames)
    return item_ids


def load_item_images(
    dataset_name: str,
    subset_name: Optional[str],
    item_ids: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Load item images as RGB uint8 arrays via subset.yml.

    All image paths are resolved through subset.yml; the Image/ directory
    is never traversed directly.

    Inputs:
        dataset_name : dataset name
        subset_name  : subset key in subset.yml; None means 'all'
        item_ids     : if provided, load only these IDs; must all exist in subset

    Outputs:
        Dict {item_id: (H, W, 3) uint8 ndarray}

    Raises:
        KeyError if any requested item_id is not found in the subset
    """
    subset_path = Path("data") / dataset_name / "subset.yml"
    if not subset_path.exists():
        raise FileNotFoundError(f"subset.yml not found: {subset_path}")

    with open(subset_path,encoding='utf-8') as f:
        subsets = yaml.safe_load(f)

    key = subset_name if subset_name is not None else "all"
    if key not in subsets:
        raise KeyError(f"Subset '{key}' not found in {subset_path}")

    # Build filename lookup: item_id -> filename
    filenames = subsets[key]
    subset_map = {Path(fn).stem: fn for fn in filenames}

    # Determine which IDs to load
    ids_to_load = item_ids if item_ids is not None else sorted(subset_map.keys())

    # Validate all requested IDs are in the subset
    missing = [iid for iid in ids_to_load if iid not in subset_map]
    if missing:
        raise KeyError(
            f"{len(missing)} item_id(s) not found in subset '{key}': {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )

    image_dir = Path("data") / dataset_name / "Image"
    images = {}
    for iid in ids_to_load:
        p = image_dir / subset_map[iid]
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {p}")
        with Image.open(p) as img:
            images[iid] = np.array(img.convert("RGB"))

    return images


# %% Device handling

def handle_device(device: str) -> str:
    """
    Resolve device selection.

    Inputs:
        device : 'cpu', 'cuda', or 'auto'

    Outputs:
        'cpu' or 'cuda'

    Raises:
        RuntimeError if 'cuda' is requested but not available
    """
    import torch
    cuda_available = torch.cuda.is_available()
    if device == "auto":
        return "cuda" if cuda_available else "cpu"
    elif device == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA device is not available.")
        return "cuda"
    return "cpu"


# %% Array utilities

def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """
    L2-normalize each row of a 2D float array.

    Inputs:
        matrix : (N, D) float ndarray

    Outputs:
        (N, D) float ndarray with unit-norm rows
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-8)
    return matrix / norms
