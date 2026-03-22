# %% Objective
"""
prepare_embedding_cache.py

Precompute and cache CLIP embeddings for all single-item images and all
semi-composite images (one item per position, all other slots blank).

Embeddings are saved as normalized matrices + index maps in .pkl files.
Raw (unnormalized) item embeddings are also saved as a .csv for inspection.

Can be called from CLI or imported by run_session.py via
load_or_create_embedding_cache().

CLI usage:
    python prepare_embedding_cache.py --experiment_id D1_M1_pairwise_config1
    python prepare_embedding_cache.py --experiment_id D1_M1_pairwise_config1 --device cuda
"""

# %% Imports
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from helper.general import load_configs, load_item_ids, load_item_images, handle_device, normalize_rows
from helper.generate_composite_image import infer_layout_patch_params, generate_semi_composite
from helper.extract_clip import create_clip_model, extract_embeddings


# %% Embedding helpers

def _batch_embed(
    images: List[np.ndarray],
    model,
    preprocess,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """
    Extract embeddings for a list of images in batches.

    Inputs:
        images     : list of (H, W, 3) uint8 ndarrays
        model      : open_clip model
        preprocess : open_clip transform
        batch_size : images per batch
        device     : 'cpu' or 'cuda'

    Outputs:
        (N, D) float32 ndarray of raw (unnormalized) embeddings
    """
    all_embs = []
    for start in tqdm(range(0, len(images), batch_size), desc="Extracting embeddings"):
        batch = images[start: start + batch_size]
        embs = extract_embeddings(batch, model, preprocess, device=device)
        all_embs.append(embs)
    return np.concatenate(all_embs, axis=0)


# %% Item embedding cache

def build_item_embedding_cache(
    experiment_id: str,
    model,
    preprocess,
    batch_size: int,
    device: str,
    item_images: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Compute and save normalized item embeddings.

    Also saves raw embeddings to item_embeddings_original.csv for inspection.

    Inputs:
        experiment_id : experiment key (determines output directory)
        model         : open_clip model
        preprocess    : open_clip transform
        batch_size    : embedding batch size
        device        : 'cpu' or 'cuda'
        item_images   : {item_id: uint8 array}

    Outputs:
        (item_embedding_matrix [N, D], item_index_map {item_id: row})
    """
    cache_dir = Path("experiment") / experiment_id / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "item.pkl"

    if cache_path.exists():
        print(f"Item cache already exists, skipping: {cache_path}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["matrix"], data["index_map"]

    item_ids = sorted(item_images.keys())
    images_ordered = [item_images[iid] for iid in item_ids]

    print(f"Extracting item embeddings for {len(item_ids)} items...")
    raw_matrix = _batch_embed(images_ordered, model, preprocess, batch_size, device)

    # Save raw embeddings as CSV for inspection
    csv_path = cache_dir / "item_embeddings_original.csv"
    D = raw_matrix.shape[1]
    emb_cols = [f"emb_dim{i+1}" for i in range(D)]
    df_raw = pd.DataFrame(raw_matrix, columns=emb_cols)
    df_raw.insert(0, "item_id", item_ids)
    df_raw.to_csv(csv_path, index=False)
    print(f"  Saved raw embeddings to {csv_path}")

    norm_matrix = normalize_rows(raw_matrix)
    index_map = {iid: i for i, iid in enumerate(item_ids)}

    with open(cache_path, "wb") as f:
        pickle.dump({"matrix": norm_matrix, "index_map": index_map}, f)
    print(f"  Saved normalized item cache to {cache_path}")

    return norm_matrix, index_map


# %% Semi-composite embedding cache

def build_semi_embedding_cache(
    experiment_id: str,
    task: str,
    model,
    preprocess,
    batch_size: int,
    device: str,
    item_images: Dict[str, np.ndarray],
    geometry,
) -> Tuple[np.ndarray, Dict[Tuple, int]]:
    """
    Compute and save normalized semi-composite embeddings for all (item, position) pairs.

    Inputs:
        experiment_id : experiment key (determines output directory)
        task          : 'pairwise' or 'grid'
        model         : open_clip model
        preprocess    : open_clip transform
        batch_size    : embedding batch size
        device        : 'cpu' or 'cuda'
        item_images   : {item_id: uint8 array}
        geometry      : LayoutGeometry

    Outputs:
        (semi_embedding_matrix [N_items * N_positions, D], semi_index_map {(item_id, pos): row})
    """
    cache_dir = Path("experiment") / experiment_id / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{task}_semi.pkl"

    if cache_path.exists():
        print(f"Semi cache already exists, skipping: {cache_path}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["matrix"], data["index_map"]

    item_ids = sorted(item_images.keys())
    positions = list(range(1, geometry.n_positions + 1))

    all_embeddings = []
    semi_index_map = {}
    row_idx = 0

    for pos in positions:
        print(f"  Extracting semi-composite embeddings for position {pos}/{geometry.n_positions}...")
        semi_images = []
        semi_keys = []

        for iid in item_ids:
            semi_arr = generate_semi_composite(item_images[iid], pos, geometry)
            semi_images.append(semi_arr)
            semi_keys.append((iid, pos))

        embs = _batch_embed(semi_images, model, preprocess, batch_size, device)
        norm_embs = normalize_rows(embs)
        all_embeddings.append(norm_embs)

        for key in semi_keys:
            semi_index_map[key] = row_idx
            row_idx += 1

        del semi_images  # release memory after each position

    semi_matrix = np.concatenate(all_embeddings, axis=0)

    with open(cache_path, "wb") as f:
        pickle.dump({"matrix": semi_matrix, "index_map": semi_index_map}, f)
    print(f"  Saved semi cache ({semi_matrix.shape[0]} entries) to {cache_path}")

    return semi_matrix, semi_index_map


# %% Load or create (used by run_session.py)

def load_or_create_embedding_cache(
    experiment_id: str,
    task: str,
    model,
    preprocess,
    batch_size: int,
    device: str,
    geometry,
    dataset_name: str,
    subset_name: str,
) -> Tuple[np.ndarray, Dict, np.ndarray, Dict]:
    """
    Load item and semi-composite embedding caches, creating them if absent.

    Inputs:
        experiment_id : experiment key
        task          : 'pairwise' or 'grid'
        model         : open_clip model
        preprocess    : open_clip transform
        batch_size    : embedding batch size
        device        : 'cpu' or 'cuda'
        geometry      : LayoutGeometry
        dataset_name  : dataset name
        subset_name   : subset key in subset.yml

    Outputs:
        (item_matrix, item_index_map, semi_matrix, semi_index_map)
    """
    item_images = load_item_images(dataset_name, subset_name)

    item_matrix, item_index_map = build_item_embedding_cache(
        experiment_id, model, preprocess, batch_size, device, item_images
    )
    semi_matrix, semi_index_map = build_semi_embedding_cache(
        experiment_id, task, model, preprocess, batch_size, device, item_images, geometry
    )

    return item_matrix, item_index_map, semi_matrix, semi_index_map


# %% CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute CLIP embedding caches for items and semi-composites."
    )
    parser.add_argument("--experiment_id", required=True,
                        help="Key in configs/experiments.yml")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"],
                        help="Compute device (default: auto)")
    return parser.parse_args()


# %% Main

def main():
    args = parse_args()

    device = handle_device(args.device)
    print(f"Using device: {device}")

    exp_cfg, model_cfg, dataset_cfg, setup_cfg = load_configs(args.experiment_id)

    task = exp_cfg["experiment"]["task"]
    dataset_name = dataset_cfg["name"]
    subset_name = dataset_cfg["subset"]
    batch_size = setup_cfg["run"]["batch_size"]
    item_patchN = setup_cfg["layout"].get("item_patchN")
    padding_patchN = setup_cfg["layout"].get("padding_patchN")

    model_spec = model_cfg["model_spec"]
    pretrained = model_cfg.get("pretrained")

    # Load base model (no forced size) to discover patch/image sizes
    model, preprocess, model_config = create_clip_model(
        model_spec=model_spec,
        pretrained=pretrained,
        device="cpu", #since this is only used to discover the patch/image sizes
    )

    # Infer layout geometry
    geometry = infer_layout_patch_params(
        task=task,
        patch_size=model_config["patch_size"],
        image_size=model_config["image_size"],
        item_patchN=item_patchN,
        padding_patchN=padding_patchN,
    )
    del model, preprocess, model_config

    print(f"Layout: overall_patchN={geometry.overall_patchN}, "
          f"item_patchN={geometry.item_patchN}, "
          f"composite pixel size={geometry.overall_pixel_size}")

    # Reload model with forced composite image size for correct patch alignment
    model, preprocess, _ = create_clip_model(
        model_spec=model_spec,
        pretrained=pretrained,
        force_image_size=geometry.overall_pixel_size,
        device=device,
    )

    item_images = load_item_images(dataset_name, subset_name)

    build_item_embedding_cache(
        args.experiment_id, model, preprocess, batch_size, device, item_images
    )
    build_semi_embedding_cache(
        args.experiment_id, task, model, preprocess, batch_size, device, item_images, geometry
    )

    print("Embedding cache preparation complete.")


if __name__ == "__main__":
    main()
