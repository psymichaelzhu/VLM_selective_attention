# %% Objective
"""
validate_attention.py

Visualize attention extraction for the first 3 trials of an experiment session.
Produces a figure with 3 rows (trials) x 3 columns:
    col 1 : raw composite image
    col 2 : attention map overlay  (viridis, alpha=0.5)
    col 3 : attention rollout overlay (viridis, alpha=0.5)

Uses the exact same model loading, geometry, and metric config as run_session.py,
so the output directly validates what the experiment pipeline is computing.

CLI usage:
    python validate_attention.py --experiment_id D1_M1_pairwise_config1
    python validate_attention.py --experiment_id D1_M1_pairwise_config1 --session_id 0
    python validate_attention.py --experiment_id D1_M1_pairwise_config1 --session_id 0 --device cuda
    python validate_attention.py --experiment_id D1_M1_pairwise_config1 --n_trials 5
"""

# %% Imports
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# script/ is the package root — add it to sys.path so that
# `from helper.x import ...` works regardless of the working directory.
ROOT = Path(__file__).resolve().parent   # .../project/script/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# All relative paths in the codebase (configs/, data/, experiment/) are
# expressed relative to project_root (one level above script/).
# Changing cwd here means every Path("configs/...") call in helper modules
# resolves correctly without modifying those files.
os.chdir(ROOT.parent)

from helper.general import load_configs, load_item_images, handle_device
from helper.generate_composite_image import (
    infer_layout_patch_params,
    generate_composite,
    LayoutGeometry,
)
from helper.extract_clip import (
    create_clip_model,
    extract_clip,
    compute_attention_map_from_storage,
    compute_attention_rollout_from_storage,
)


# %% Attention → spatial heatmap

def attention_to_heatmap(
    attn_1d: np.ndarray,
    geometry: LayoutGeometry,
) -> np.ndarray:
    """
    Reshape a flat (n_patches,) attention vector into a 2-D spatial heatmap
    and upsample it to composite pixel size via nearest-neighbour.

    Inputs:
        attn_1d  : (n_patches,) float array, where n_patches = overall_patchN^2
        geometry : LayoutGeometry

    Outputs:
        (overall_pixel_size, overall_pixel_size) float32 array in [0, 1]
    """
    opN = geometry.overall_patchN
    ps  = geometry.patch_size

    # Reshape to patch grid
    heatmap_patches = attn_1d.reshape(opN, opN)          # (opN, opN)

    # Upsample each patch to patch_size pixels (nearest-neighbour = no blur)
    heatmap_pixels = np.repeat(np.repeat(heatmap_patches, ps, axis=0), ps, axis=1)
    # → (overall_pixel_size, overall_pixel_size)

    # Normalise to [0, 1] for colourmap
    lo, hi = heatmap_pixels.min(), heatmap_pixels.max()
    if hi > lo:
        heatmap_pixels = (heatmap_pixels - lo) / (hi - lo)
    else:
        heatmap_pixels = np.zeros_like(heatmap_pixels)

    return heatmap_pixels.astype(np.float32)


def overlay_heatmap(
    composite: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    cmap_name: str = "viridis",
) -> np.ndarray:
    """
    Blend a normalised 2-D heatmap over an RGB composite image.

    Inputs:
        composite : (H, W, 3) uint8 image
        heatmap   : (H, W) float32 in [0, 1]
        alpha     : heatmap opacity
        cmap_name : matplotlib colourmap name

    Outputs:
        (H, W, 3) uint8 blended image
    """
    cmap   = cm.get_cmap(cmap_name)
    colour = cmap(heatmap)[..., :3]                         # (H, W, 3) float [0,1]
    base   = composite.astype(np.float32) / 255.0           # (H, W, 3) float [0,1]
    blended = (1.0 - alpha) * base + alpha * colour
    return (blended * 255).clip(0, 255).astype(np.uint8)


# %% First-attention-metric config resolution

def resolve_attention_configs(metrics_cfg: list) -> dict:
    """
    Extract the first attention_map and attention_rollout metric configs.

    Returns a dict with keys 'attention_map' and/or 'attention_rollout',
    each holding the raw config dict from setups.yml.

    If neither is present, raises ValueError — this script needs at least one.
    """
    resolved = {}
    for m in metrics_cfg:
        metric = m["metric"]
        if metric == "attention_map" and "attention_map" not in resolved:
            resolved["attention_map"] = m
        elif metric == "attention_rollout" and "attention_rollout" not in resolved:
            resolved["attention_rollout"] = m
        if len(resolved) == 2:
            break

    if not resolved:
        raise ValueError(
            "No attention_map or attention_rollout metrics found in setup config. "
            "This script requires at least one attention metric to be configured."
        )
    return resolved


# %% Main visualisation

def validate_attention(
    experiment_id: str,
    session_id: int = 0,
    n_trials: int = 3,
    device: str = "auto",
    output_path: str = None,
) -> None:
    """
    Load configs, run CLIP on the first n_trials of a session, and plot the
    3-column attention validation figure.

    Inputs:
        experiment_id : key in configs/experiments.yml
        session_id    : which session's design matrix to read (default 0)
        n_trials      : how many trials to show (default 3)
        device        : 'cpu', 'cuda', or 'auto'
        output_path   : if provided, save figure here; otherwise show interactively
    """
    # ── 1. Load configs ──────────────────────────────────────────────────────
    exp_cfg, model_cfg, dataset_cfg, setup_cfg = load_configs(experiment_id)

    task         = exp_cfg["experiment"]["task"]
    dataset_name = dataset_cfg["name"]
    subset_name  = dataset_cfg["subset"]
    item_patchN     = setup_cfg["layout"].get("item_patchN")
    padding_patchN  = setup_cfg["layout"].get("padding_patchN")
    metrics_cfg     = setup_cfg["metrics"]
    model_spec      = model_cfg["model_spec"]
    pretrained      = model_cfg.get("pretrained")

    device = handle_device(device)
    print(f"Device: {device}")

    # Resolve attention configs from the experiment's metric list
    attn_cfgs = resolve_attention_configs(metrics_cfg)
    has_map     = "attention_map"     in attn_cfgs
    has_rollout = "attention_rollout" in attn_cfgs

    # ── Config log ───────────────────────────────────────────────────────────
    sep = "─" * 52
    print(sep)
    print(f"  experiment_id : {experiment_id}")
    print(f"  session_id    : {session_id}")
    print(f"  n_trials      : {n_trials}")
    print(f"  device        : {device}")
    print(sep)
    print(f"  model_spec    : {model_spec}")
    print(f"  pretrained    : {pretrained}")
    print(sep)
    print(f"  task          : {task}")
    print(f"  dataset       : {dataset_name}  /  subset: {subset_name}")
    print(f"  item_patchN   : {item_patchN}")
    print(f"  padding_patchN: {padding_patchN}")
    print(sep)
    if has_map:
        m = attn_cfgs["attention_map"]
        print(f"  attention_map")
        print(f"    layer             : {m['layer']}")
        print(f"    head_reduction    : {m['head_reduction']}")
        print(f"    query_token_index : {m.get('query_token_index', 0)}")
    else:
        print(f"  attention_map     : [not in config — using fallback]")
    print(sep)
    if has_rollout:
        m = attn_cfgs["attention_rollout"]
        print(f"  attention_rollout")
        print(f"    head_reduction    : {m['head_reduction']}")
        print(f"    discard_ratio     : {m.get('discard_ratio', 0.95)}")
        print(f"    query_token_index : {m.get('query_token_index', 0)}")
    else:
        print(f"  attention_rollout : [not in config — using fallback]")
    print(sep)

    # ── 2. Load session design, take first n_trials ──────────────────────────
    design_path = Path("experiment") / experiment_id / "design_matrix" / f"{session_id}.csv"
    if not design_path.exists():
        raise FileNotFoundError(f"Session design not found: {design_path}")
    session_df = pd.read_csv(design_path).head(n_trials).reset_index(drop=True)
    print(f"Loaded {len(session_df)} trials from session {session_id}")

    # ── 3. Load model (geometry discovery pass) ──────────────────────────────
    _, _, model_config = create_clip_model(
        model_spec=model_spec,
        pretrained=pretrained,
        device="cpu",           # geometry-only; same logic as run_session.py
    )

    geometry = infer_layout_patch_params(
        task=task,
        patch_size=model_config["patch_size"],
        image_size=model_config["image_size"],
        item_patchN=item_patchN,
        padding_patchN=padding_patchN,
    )
    print(f"Geometry: {geometry.overall_patchN} patches/side, "
          f"{geometry.overall_pixel_size}px composite")

    # Reload with forced composite size — exact same as run_session.py
    model, preprocess, _ = create_clip_model(
        model_spec=model_spec,
        pretrained=pretrained,
        force_image_size=geometry.overall_pixel_size,
        device=device,
    )

    # ── 4. Build composite images ─────────────────────────────────────────────
    all_item_ids = []
    for pos in range(1, geometry.n_positions + 1):
        col = f"item_{pos}"
        if col in session_df.columns:
            all_item_ids.extend(session_df[col].dropna().tolist())

    item_images = load_item_images(dataset_name, subset_name, item_ids=list(set(all_item_ids)))

    composites = []
    for _, row in session_df.iterrows():
        pos_to_img = {}
        for pos in range(1, geometry.n_positions + 1):
            iid = row.get(f"item_{pos}")
            if pd.notna(iid) and iid:
                pos_to_img[pos] = item_images[iid]
        composites.append(generate_composite(pos_to_img, geometry))

    print(f"Built {len(composites)} composite images")

    # ── 5. CLIP forward pass with attention capture ───────────────────────────
    clip_output = extract_clip(
        composites,
        model,
        preprocess,
        device=device,
        need_attention=True,   # always True for validation
    )
    attn_storage = clip_output["attn_storage"]
    print(f"Captured {len(attn_storage)} attention layer tensors")

    # ── 6. Compute attention maps and rollouts ────────────────────────────────
    # attention_map — use first configured metric's settings, or sensible defaults
    if has_map:
        m_cfg = attn_cfgs["attention_map"]
        attn_maps = compute_attention_map_from_storage(
            attn_storage,
            layer=m_cfg["layer"],
            head_reduction=m_cfg["head_reduction"],
            query_token_index=m_cfg.get("query_token_index", 0),
        )
        map_label = (
            f"attention_map  "
            f"layer={m_cfg['layer']}  "
            f"head={m_cfg['head_reduction']}"
        )
    else:
        # Fall back to last layer, mean heads — show something useful
        attn_maps = compute_attention_map_from_storage(
            attn_storage,
            layer=1.0,
            head_reduction="mean",
        )
        map_label = "attention_map  layer=1.0  head=mean  [fallback]"

    # attention_rollout — use first configured metric's settings, or defaults
    if has_rollout:
        m_cfg = attn_cfgs["attention_rollout"]
        rollouts = compute_attention_rollout_from_storage(
            attn_storage,
            head_reduction=m_cfg["head_reduction"],
            discard_ratio=m_cfg.get("discard_ratio", 0.95),
            query_token_index=m_cfg.get("query_token_index", 0),
        )
        rollout_label = (
            f"attention_rollout  "
            f"head={m_cfg['head_reduction']}  "
            f"discard={m_cfg.get('discard_ratio', 0.95)}"
        )
    else:
        rollouts = compute_attention_rollout_from_storage(
            attn_storage,
            head_reduction="mean",
            discard_ratio=0.95,
        )
        rollout_label = "attention_rollout  head=mean  discard=0.95  [fallback]"

    # attn_maps and rollouts are (B, n_patches); B = n_trials
    print(f"Attention map shape:  {attn_maps.shape}")
    print(f"Rollout shape:        {rollouts.shape}")

    # ── 7. Plot ───────────────────────────────────────────────────────────────
    n_rows = n_trials
    n_cols = 3
    fig_w  = n_cols * 3.5
    fig_h  = n_rows * 3.8

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = axes[np.newaxis, :]   # keep 2-D indexing

    col_titles = [
        "composite image",
        map_label,
        rollout_label,
    ]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=8, pad=6)

    for trial_idx in range(n_trials):
        composite = composites[trial_idx]                        # (H, W, 3) uint8
        trial_id  = session_df.loc[trial_idx, "trial_id"]

        # --- col 0: raw composite ---
        ax = axes[trial_idx, 0]
        ax.imshow(composite)
        ax.set_ylabel(f"trial {trial_id}", fontsize=8, labelpad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # --- col 1: attention map overlay ---
        attn_1d  = attn_maps[trial_idx]                         # (n_patches,)
        heatmap  = attention_to_heatmap(attn_1d, geometry)
        overlay  = overlay_heatmap(composite, heatmap, alpha=0.5, cmap_name="viridis")
        ax = axes[trial_idx, 1]
        ax.imshow(overlay)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # --- col 2: attention rollout overlay ---
        rollout_1d = rollouts[trial_idx]                        # (n_patches,)
        heatmap_r  = attention_to_heatmap(rollout_1d, geometry)
        overlay_r  = overlay_heatmap(composite, heatmap_r, alpha=0.5, cmap_name="viridis")
        ax = axes[trial_idx, 2]
        ax.imshow(overlay_r)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    # shared colourbar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    sm.set_array([])

    fig.suptitle(
        f"Attention validation — {experiment_id}  |  session {session_id}  "
        f"|  trials 0–{n_trials - 1}",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()

    # ── 8. Save or show ───────────────────────────────────────────────────────
    if output_path is None:
        out = (
            Path("experiment") / experiment_id
            / f"validate_attention_s{session_id}.png"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        output_path = str(out)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved validation figure → {output_path}")
    plt.close(fig)


# %% CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise attention extraction for the first N trials of an experiment session."
    )
    parser.add_argument("--experiment_id", required=True,
                        help="Key in configs/experiments.yml")
    parser.add_argument("--session_id", type=int, default=0,
                        help="Session ID to use (default: 0)")
    parser.add_argument("--n_trials", type=int, default=4,
                        help="Number of trials to visualise (default: 4)")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"],
                        help="Compute device (default: auto)")
    parser.add_argument("--output", default=None,
                        help="Path to save the figure (default: experiment/<id>/validate_attention_s<N>.png)")
    return parser.parse_args()


# %% Main

def main():
    args = parse_args()
    validate_attention(
        experiment_id=args.experiment_id,
        session_id=args.session_id,
        n_trials=args.n_trials,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()