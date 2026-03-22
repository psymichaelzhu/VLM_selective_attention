# %% Objective
"""
run_session.py

Run one session for a given experiment. Processes trials batch by batch and
appends results incrementally to a session-level CSV.

Supports resuming: already-processed trials (tracked by unique trial_id count)
are skipped on restart.

Execution is fully metric-driven: the metric plan is read from
setup_cfg["metrics"], and only one CLIP forward pass is performed per batch.
All requested cosine and attention metrics are derived from that single pass.

CLI usage:
    python run_session.py --experiment_id D1_M1_pairwise_config1 --session_id 0
    python run_session.py --experiment_id D1_M1_grid_config1 --session_id 3
"""

# %% Imports
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import argparse

from helper.general import load_configs, load_item_images, handle_device, normalize_rows
from helper.generate_composite_image import (
    infer_layout_patch_params,
    generate_composite,
    generate_composite_template,
    LayoutGeometry,
)
from helper.extract_clip import (
    create_clip_model,
    extract_clip,
    compute_attention_map_from_storage,
    compute_attention_rollout_from_storage,
)
from prepare_embedding_cache import load_or_create_embedding_cache


# %% Session loading and resume logic

def load_session_design(experiment_id: str, session_id: int) -> pd.DataFrame:
    """
    Load session design matrix from disk.

    Inputs:
        experiment_id : experiment key
        session_id    : integer session ID

    Outputs:
        DataFrame of trials for this session
    """
    path = Path("experiment") / experiment_id / "design_matrix" / f"{session_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Session design not found: {path}")
    return pd.read_csv(path)


def count_completed_trials(output_csv: Path) -> int:
    """
    Count unique trial_ids already written to the output CSV.

    Does not rely on a fixed number of rows per trial; reads trial_id column only.

    Inputs:
        output_csv : path to output CSV (may not exist yet)

    Outputs:
        Number of unique trial_ids already written
    """
    if not output_csv.exists():
        return 0
    df = pd.read_csv(output_csv, usecols=["trial_id"])
    return df["trial_id"].nunique()


# %% Composite image generation for a batch

def build_composite_batch(
    batch_df: pd.DataFrame,
    item_images: Dict[str, np.ndarray],
    geometry: LayoutGeometry,
) -> List[np.ndarray]:
    """
    Generate composite images for all trials in a batch.

    Inputs:
        batch_df    : subset of design matrix
        item_images : {item_id: uint8 array}
        geometry    : LayoutGeometry

    Outputs:
        List of (overall_pixel_size, overall_pixel_size, 3) uint8 arrays
    """
    n_positions = geometry.n_positions
    composites = []
    for _, row in batch_df.iterrows():
        pos_to_img = {}
        for pos in range(1, n_positions + 1):
            iid = row.get(f"item_{pos}")
            if pd.notna(iid) and iid:
                pos_to_img[pos] = item_images[iid]
        composites.append(generate_composite(pos_to_img, geometry))
    return composites


# %% Attention summarization per position

def summarize_attention_per_position(
    attention: np.ndarray,
    template: Dict[int, List[int]],
    geometry: LayoutGeometry,
    summaries: List[str],
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Compute requested summary statistics of attention per position per trial.

    Inputs:
        attention  : (B, n_patches) attention array
        template   : {position: [patch_indices]}
        geometry   : LayoutGeometry
        summaries  : list of summary types to compute, e.g. ['mean', 'max']

    Outputs:
        Dict {summary_type: {position: (B,) array}}
        e.g. {'mean': {1: array, 2: array}, 'max': {1: array, 2: array}}
    """
    result: Dict[str, Dict[int, np.ndarray]] = {s: {} for s in summaries}

    for pos in range(1, geometry.n_positions + 1):
        patch_idx = template[pos]
        patch_attn = attention[:, patch_idx]   # (B, n_item_patches^2)
        if "mean" in summaries:
            result["mean"][pos] = patch_attn.mean(axis=1)
        if "max" in summaries:
            result["max"][pos] = patch_attn.max(axis=1)

    return result


# %% Similarity computation

def compute_similarities(
    composite_embs: np.ndarray,
    batch_df: pd.DataFrame,
    geometry: LayoutGeometry,
    matrix: np.ndarray,
    index_map: Dict,
    mode: str,
) -> Dict[int, np.ndarray]:
    """
    Compute dot-product similarity between each composite embedding and a
    precomputed embedding matrix (either item or semi-composite).

    Inputs:
        composite_embs : (B, D) normalized composite embeddings
        batch_df       : batch design matrix
        geometry       : LayoutGeometry
        matrix         : (N, D) normalized embedding matrix
        index_map      : {item_id: row} if mode='item';
                         {(item_id, pos): row} if mode='semi'
        mode           : 'item' or 'semi'

    Outputs:
        Dict {position: (B,) similarity array}; NaN where position is empty.
    """
    if mode not in ("item", "semi"):
        raise ValueError(f"mode must be 'item' or 'semi', got '{mode}'")

    B = len(batch_df)
    sims = {pos: np.full(B, np.nan) for pos in range(1, geometry.n_positions + 1)}

    for pos in range(1, geometry.n_positions + 1):
        col = f"item_{pos}"
        if col not in batch_df.columns:
            continue
        item_ids = batch_df[col].values

        valid_mask = pd.notna(item_ids)
        if not valid_mask.any():
            continue

        if mode == "semi":
            keys = [(iid, pos) for iid in item_ids[valid_mask]]
        else:
            keys = list(item_ids[valid_mask])

        gathered_rows = [index_map[k] for k in keys]
        gathered = matrix[gathered_rows]                             # (k, D)
        dot = (composite_embs[valid_mask] * gathered).sum(axis=1)   # (k,)
        sims[pos][valid_mask] = dot

    return sims


# %% Metric planning

def plan_metrics(
    metrics_cfg: List[dict],
) -> Tuple[bool, bool]:
    """
    Inspect the metric config list and determine which base outputs are needed.

    Inputs:
        metrics_cfg : list of metric config dicts from setup_cfg["metrics"]

    Outputs:
        (need_cosine, need_attention)
        need_cosine    : True if any cosine_similarity metric is requested
        need_attention : True if any attention_map or attention_rollout is requested
    """
    need_cosine = any(m["metric"] == "cosine_similarity" for m in metrics_cfg)
    need_attention = any(m["metric"] in ("attention_map", "attention_rollout") for m in metrics_cfg)
    return need_cosine, need_attention


# %% Results assembly

def assemble_batch_results(
    batch_df: pd.DataFrame,
    geometry: LayoutGeometry,
    metric_blocks: List[Tuple[str, str, Dict[int, np.ndarray]]],
) -> pd.DataFrame:
    """
    Assemble per-trial metric results into a long-format DataFrame.

    Schema: trial_id, metric, variant, value1 [... valueN]
    Number of value columns equals geometry.n_positions (2 or 9).

    Inputs:
        batch_df      : batch design matrix (for trial_ids)
        geometry      : LayoutGeometry
        metric_blocks : list of (metric, variant, {position: (B,) array})

    Outputs:
        DataFrame in long metric format
    """
    n_positions = geometry.n_positions
    trial_ids = batch_df["trial_id"].values
    value_cols = [f"value{i}" for i in range(1, n_positions + 1)]

    def _make_rows(sims_dict: Dict[int, np.ndarray], metric: str, variant: str) -> pd.DataFrame:
        matrix = np.column_stack([sims_dict[p] for p in range(1, n_positions + 1)])
        rows = pd.DataFrame(matrix, columns=value_cols)
        rows.insert(0, "trial_id", trial_ids)
        rows.insert(1, "metric", metric)
        rows.insert(2, "variant", variant)
        return rows

    frames = [_make_rows(sims, metric, variant) for metric, variant, sims in metric_blocks]
    return pd.concat(frames, ignore_index=True)


def append_batch_results(batch_result_df: pd.DataFrame, output_csv: Path) -> None:
    """
    Append batch results to the session output CSV.

    Inputs:
        batch_result_df : assembled result DataFrame
        output_csv      : path to output CSV
    """
    write_header = not output_csv.exists()
    batch_result_df.to_csv(output_csv, mode="a", header=write_header, index=False)


# %% Batch runner

def run_batch(
    batch_df: pd.DataFrame,
    task: str,
    dataset_name: str,
    subset_name: str,
    model,
    preprocess,
    device: str,
    geometry: LayoutGeometry,
    template: Dict[int, List[int]],
    item_matrix: np.ndarray,
    item_index_map: Dict,
    semi_matrix: np.ndarray,
    semi_index_map: Dict,
    setup_cfg: dict,
    output_csv: Path,
) -> None:
    """
    Process one batch of trials end-to-end and append results to CSV.

    Determines required base outputs from setup_cfg["metrics"], performs one
    CLIP forward pass, then derives all configured metrics from that pass.

    Inputs:
        batch_df        : subset of session design matrix
        task            : 'pairwise' or 'grid'
        dataset_name    : dataset name
        subset_name     : subset key in subset.yml
        model           : open_clip model (monkey-patched for attention)
        preprocess      : open_clip transform
        device          : 'cpu' or 'cuda'
        geometry        : LayoutGeometry
        template        : {position: [patch_indices]}
        item_matrix     : (N_items, D) normalized item embeddings
        item_index_map  : {item_id: row}
        semi_matrix     : (N_items * N_positions, D) normalized semi embeddings
        semi_index_map  : {(item_id, pos): row}
        setup_cfg       : setup config dict (contains "metrics")
        output_csv      : path to output CSV
    """
    metrics_cfg = setup_cfg["metrics"]
    _, need_attention = plan_metrics(metrics_cfg)

    # 1. Collect all unique item_ids needed for this batch
    all_item_ids = []
    for pos in range(1, geometry.n_positions + 1):
        col = f"item_{pos}"
        if col in batch_df.columns:
            all_item_ids.extend(batch_df[col].dropna().tolist())

    # 2. Load images for this batch
    item_images = load_item_images(dataset_name, subset_name, item_ids=list(set(all_item_ids)))

    # 3. Generate composite images
    composites = build_composite_batch(batch_df, item_images, geometry)
    del item_images

    # 4. One CLIP forward pass — capture attention storage only if needed
    clip_output = extract_clip(
        composites,
        model,
        preprocess,
        device=device,
        need_attention=need_attention,
    )
    del composites

    raw_embs = clip_output["embeddings"]       # (B, D)
    attn_storage = clip_output["attn_storage"] # list of (block_idx, tensor) or []

    # 5. Normalize composite embeddings
    composite_embs = normalize_rows(raw_embs)
    del raw_embs

    # 6. Build result blocks from metric config
    metric_blocks: List[Tuple[str, str, Dict[int, np.ndarray]]] = []

    for m_cfg in metrics_cfg:
        metric = m_cfg["metric"]
        variant = m_cfg["variant"]

        if metric == "cosine_similarity":
            mode = variant  # 'item' or 'semi'
            if mode == "item":
                sims = compute_similarities(
                    composite_embs, batch_df, geometry,
                    item_matrix, item_index_map, mode="item",
                )
            elif mode == "semi":
                sims = compute_similarities(
                    composite_embs, batch_df, geometry,
                    semi_matrix, semi_index_map, mode="semi",
                )
            else:
                raise ValueError(f"Unknown cosine_similarity variant: '{variant}'")
            metric_blocks.append((metric, variant, sims))

        elif metric == "attention_map":
            attn = compute_attention_map_from_storage(
                attn_storage,
                layer=m_cfg["layer"],
                head_reduction=m_cfg["head_reduction"],
                query_token_index=m_cfg.get("query_token_index", 0),
            )
            summaries = m_cfg.get("summary", ["mean", "max"])
            summary_results = summarize_attention_per_position(attn, template, geometry, summaries)
            for s in summaries:
                metric_blocks.append((metric, f"{variant}__{s}", summary_results[s]))

        elif metric == "attention_rollout":
            rollout = compute_attention_rollout_from_storage(
                attn_storage,
                head_reduction=m_cfg["head_reduction"],
                discard_ratio=m_cfg.get("discard_ratio", 0.95),
                query_token_index=m_cfg.get("query_token_index", 0),
            )
            summaries = m_cfg.get("summary", ["mean", "max"])
            summary_results = summarize_attention_per_position(rollout, template, geometry, summaries)
            for s in summaries:
                metric_blocks.append((metric, f"{variant}__{s}", summary_results[s]))

        else:
            raise ValueError(f"Unknown metric: '{metric}'")

    del composite_embs, attn_storage

    # 7. Assemble and save results
    batch_result_df = assemble_batch_results(batch_df, geometry, metric_blocks)
    append_batch_results(batch_result_df, output_csv)


# %% Session runner

def run_session(
    experiment_id: str,
    session_id: int,
    device: str,
) -> None:
    """
    Run all remaining batches in one session.

    Inputs:
        experiment_id : experiment key
        session_id    : integer session ID
        device        : 'cpu' or 'cuda'
    """
    exp_cfg, model_cfg, dataset_cfg, setup_cfg = load_configs(experiment_id)

    task = exp_cfg["experiment"]["task"]
    dataset_name = dataset_cfg["name"]
    subset_name = dataset_cfg["subset"]
    batch_size = setup_cfg["run"]["batch_size"]
    item_patchN = setup_cfg["layout"].get("item_patchN")
    padding_patchN = setup_cfg["layout"].get("padding_patchN")

    model_spec = model_cfg["model_spec"]
    pretrained = model_cfg.get("pretrained")

    # Output path
    output_dir = Path("experiment") / experiment_id / "experiment_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{session_id}.csv"

    # Load session design
    session_df = load_session_design(experiment_id, session_id)
    n_total = len(session_df)
    n_done = count_completed_trials(output_csv)
    print(f"Session {session_id}: {n_total} trials total, {n_done} already completed.")

    if n_done >= n_total:
        print("Session already complete. Exiting.")
        return

    # Load base model (no forced size) to discover patch/image sizes
    _, _, model_config = create_clip_model(
        model_spec=model_spec,
        pretrained=pretrained,
        device=device,
    )

    # Infer layout geometry
    geometry = infer_layout_patch_params(
        task=task,
        patch_size=model_config["patch_size"],
        image_size=model_config["image_size"],
        item_patchN=item_patchN,
        padding_patchN=padding_patchN,
    )
    print(f"Composite size: {geometry.overall_pixel_size}px "
          f"({geometry.overall_patchN} patches/side)")

    # Reload model with forced composite image size for correct patch alignment
    # Monkey-patching is applied once inside create_clip_model()
    model, preprocess, _ = create_clip_model(
        model_spec=model_spec,
        pretrained=pretrained,
        force_image_size=geometry.overall_pixel_size,
        device=device,
    )

    # Generate patch-level composite template (once per session)
    template = generate_composite_template(geometry)

    # Load or create embedding caches
    item_matrix, item_index_map, semi_matrix, semi_index_map = load_or_create_embedding_cache(
        experiment_id=experiment_id,
        task=task,
        model=model,
        preprocess=preprocess,
        batch_size=batch_size,
        device=device,
        geometry=geometry,
        dataset_name=dataset_name,
        subset_name=subset_name,
    )

    # Process remaining trials
    remaining_df = session_df.iloc[n_done:].reset_index(drop=True)
    n_remaining = len(remaining_df)
    n_batches = (n_remaining + batch_size - 1) // batch_size
    print(f"Running {n_remaining} remaining trials in {n_batches} batches...")

    for batch_idx in tqdm(range(n_batches), desc="Batches"):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_remaining)
        batch_df = remaining_df.iloc[start:end]

        run_batch(
            batch_df=batch_df,
            task=task,
            dataset_name=dataset_name,
            subset_name=subset_name,
            model=model,
            preprocess=preprocess,
            device=device,
            geometry=geometry,
            template=template,
            item_matrix=item_matrix,
            item_index_map=item_index_map,
            semi_matrix=semi_matrix,
            semi_index_map=semi_index_map,
            setup_cfg=setup_cfg,
            output_csv=output_csv,
        )

    print(f"Session {session_id} complete. Results saved to {output_csv}")


# %% CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one session for a given experiment."
    )
    parser.add_argument("--experiment_id", required=True,
                        help="Key in configs/experiments.yml")
    parser.add_argument("--session_id", type=int, required=True,
                        help="Integer session ID to process")
    return parser.parse_args()


# %% Main

def main():
    args = parse_args()
    device = handle_device("auto")
    run_session(
        experiment_id=args.experiment_id,
        session_id=args.session_id,
        device=device,
    )


if __name__ == "__main__":
    main()
