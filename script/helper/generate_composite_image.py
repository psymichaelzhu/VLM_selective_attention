# %% Objective
"""
helper/generate_composite_image.py

Generate composite and semi-composite images and patch-level position templates
for pairwise and grid tasks. Called internally by run_session.py and
prepare_embedding_cache.py — not run independently.

All composite construction uses NumPy array operations (no PIL paste).
"""

# %% Imports
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# %% Layout geometry

@dataclass
class LayoutGeometry:
    """All pixel- and patch-level dimensions for one layout configuration."""
    task: str
    patch_size: int
    image_size: int              # original model image size (e.g. 224)
    item_patchN: int
    padding_patchN: int
    overall_patchN: int          # total patches per side of composite
    overall_pixel_size: int      # overall_patchN * patch_size
    item_pixel_size: int
    padding_pixel_size: int
    left_margin: int             # pixels; may be 0 if item_patchN was specified
    top_margin: int
    n_positions: int             # 2 for pairwise, 9 for grid


def infer_layout_patch_params(
    task: str,
    patch_size: int,
    image_size: int,
    item_patchN: Optional[int] = None,
    padding_patchN: Optional[int] = None,
) -> LayoutGeometry:
    """
    Infer full layout geometry from task + model dimensions.

    If item_patchN and padding_patchN are provided, overall size is derived.
    Otherwise, they are inferred to maximally fill image_size.

    Inputs:
        task            : 'pairwise' or 'grid'
        patch_size      : model patch size in pixels
        image_size      : model input image size in pixels
        item_patchN     : patches per item side (optional)
        padding_patchN  : patches for gap between items (optional)

    Outputs:
        LayoutGeometry dataclass
    """
    if task not in ("pairwise", "grid"):
        raise ValueError(f"task must be 'pairwise' or 'grid', got '{task}'")

    specified = (item_patchN is not None) and (padding_patchN is not None)

    if specified:
        if task == "pairwise":
            overall_patchN = 2 * item_patchN + 1 * padding_patchN
        else:  # grid
            overall_patchN = 3 * item_patchN + 2 * padding_patchN
        left_margin = 0
        top_margin = 0

    else:
        overall_patchN = image_size // patch_size
        padding_patchN = 1

        if task == "pairwise":
            if overall_patchN < 3:
                raise ValueError(
                    f"image_size={image_size} too small for pairwise layout "
                    f"with patch_size={patch_size}"
                )
            item_patchN = (overall_patchN - padding_patchN) // 2
            remainder = overall_patchN - (2 * item_patchN + padding_patchN)

        else:  # grid
            if overall_patchN < 5:
                raise ValueError(
                    f"image_size={image_size} too small for grid layout "
                    f"with patch_size={patch_size}"
                )
            item_patchN = (overall_patchN - 2 * padding_patchN) // 3
            remainder = overall_patchN - (3 * item_patchN + 2 * padding_patchN)

        left_margin = remainder // 2
        top_margin = remainder - left_margin

    overall_pixel_size = overall_patchN * patch_size
    item_pixel_size = item_patchN * patch_size
    padding_pixel_size = padding_patchN * patch_size
    n_positions = 2 if task == "pairwise" else 9

    return LayoutGeometry(
        task=task,
        patch_size=patch_size,
        image_size=image_size,
        item_patchN=item_patchN,
        padding_patchN=padding_patchN,
        overall_patchN=overall_patchN,
        overall_pixel_size=overall_pixel_size,
        item_pixel_size=item_pixel_size,
        padding_pixel_size=padding_pixel_size,
        left_margin=left_margin * patch_size,
        top_margin=top_margin * patch_size,
        n_positions=n_positions,
    )


def compute_position_offsets(geometry: LayoutGeometry) -> Dict[int, Tuple[int, int]]:
    """
    Compute top-left pixel offsets for each position index.

    Inputs:
        geometry : LayoutGeometry

    Outputs:
        Dict mapping position index (1-based) to (row_px, col_px) top-left offset.
        Pairwise: positions 1, 2 (left, right).
        Grid: positions 1–9, row-major.
    """
    g = geometry
    offsets = {}

    if g.task == "pairwise":
        row_px = (g.overall_pixel_size - g.item_pixel_size) // 2
        for col_idx, pos in enumerate([1, 2]):
            col_px = g.left_margin + col_idx * (g.item_pixel_size + g.padding_pixel_size)
            offsets[pos] = (row_px, col_px)

    else:  # grid
        for row_idx in range(3):
            for col_idx in range(3):
                pos = row_idx * 3 + col_idx + 1  # 1-based, row-major
                row_px = g.top_margin + row_idx * (g.item_pixel_size + g.padding_pixel_size)
                col_px = g.left_margin + col_idx * (g.item_pixel_size + g.padding_pixel_size)
                offsets[pos] = (row_px, col_px)

    return offsets


# %% Composite image generation

def _make_canvas(geometry: LayoutGeometry) -> np.ndarray:
    """Return a white canvas (3 channels) of overall_pixel_size x overall_pixel_size."""
    size = geometry.overall_pixel_size
    return np.ones((size, size, 3), dtype=np.uint8) * 255


def _paste_item(
    canvas: np.ndarray,
    item_arr: np.ndarray,
    row_px: int,
    col_px: int,
    item_pixel_size: int,
) -> None:
    """
    Resize item array to item_pixel_size and paste in-place onto canvas.

    Inputs:
        canvas          : (H, W, 3) uint8 array, modified in place
        item_arr        : (H', W', 3) uint8 array
        row_px, col_px  : top-left corner
        item_pixel_size : target size in pixels
    """
    img = Image.fromarray(item_arr)
    img = img.resize((item_pixel_size, item_pixel_size), Image.BILINEAR)
    arr = np.array(img)
    canvas[row_px: row_px + item_pixel_size, col_px: col_px + item_pixel_size] = arr


def generate_composite(
    position_to_image: Dict[int, np.ndarray],
    geometry: LayoutGeometry,
) -> np.ndarray:
    """
    Place items at specified positions on a white canvas.

    Works for both pairwise (keys 1, 2) and grid (keys 1–9) layouts.
    Positions not present in position_to_image remain white (blank).

    Inputs:
        position_to_image : dict mapping position index (1-based) to (H, W, 3) uint8 array
        geometry          : LayoutGeometry

    Outputs:
        (overall_pixel_size, overall_pixel_size, 3) uint8 composite array
    """
    offsets = compute_position_offsets(geometry)
    canvas = _make_canvas(geometry)
    for pos, item_arr in position_to_image.items():
        _paste_item(canvas, item_arr, *offsets[pos], geometry.item_pixel_size)
    return canvas


def generate_semi_composite(
    item_img: np.ndarray,
    position: int,
    geometry: LayoutGeometry,
) -> np.ndarray:
    """
    Place a single item at a given position on a white canvas; all other slots blank.

    Inputs:
        item_img : (H, W, 3) uint8
        position : 1-based position index
        geometry : LayoutGeometry

    Outputs:
        (overall_pixel_size, overall_pixel_size, 3) uint8 array
    """
    return generate_composite({position: item_img}, geometry)


# %% Patch-level position templates

def _position_patch_indices(
    position: int,
    geometry: LayoutGeometry,
) -> List[int]:
    """
    Return flat patch indices (0-based, CLS excluded) for a given position.

    Patch grid is overall_patchN x overall_patchN; index 0 is top-left,
    scanning row-major.

    Inputs:
        position : 1-based position index
        geometry : LayoutGeometry

    Outputs:
        List of integer patch indices
    """
    offsets = compute_position_offsets(geometry)
    row_px, col_px = offsets[position]
    ps = geometry.patch_size
    opN = geometry.overall_patchN

    row_start = row_px // ps
    col_start = col_px // ps
    n = geometry.item_patchN

    indices = []
    for r in range(row_start, row_start + n):
        for c in range(col_start, col_start + n):
            indices.append(r * opN + c)
    return indices


def generate_composite_template(geometry: LayoutGeometry) -> Dict[int, List[int]]:
    """
    Build patch-index template for all positions in a layout.

    Inputs:
        geometry : LayoutGeometry

    Outputs:
        Dict {position_index: [patch_indices]} for all positions in the layout
    """
    positions = range(1, geometry.n_positions + 1)
    return {pos: _position_patch_indices(pos, geometry) for pos in positions}
