# %% Objective
"""
helper/extract_clip.py

1. Load a CLIP ViT model via open_clip (supports built-in, hf-hub:, local-dir: specs).
2. Attach persistent attention storage via one-time monkey-patching of transformer blocks.
3. Extract embeddings and raw attention storage in a single forward pass.
4. Derive attention maps and rollouts from the stored tensors via reusable helpers.

Attention capture:
    - Transformer blocks are monkey-patched once during model creation.
    - Before each attention-enabled forward, call _clear_attention_storage(model).
    - After the forward, call _get_attention_storage(model) to retrieve tensors.
    - Each stored tensor is shaped (B, H, N, N).

Layer selection:
    - All attention metric configs specify `layer` as relative depth in [0, 1].
    - 0.0 = first transformer block, 1.0 = last, 0.5 = middle.
    - Mapped to nearest integer index via: round(layer * (n_layers - 1)).

Called internally by run_session.py and prepare_embedding_cache.py.
"""

# %% Imports
import numpy as np
import torch
import open_clip
from typing import List, Optional, Tuple

from helper.general import handle_device


# %% Model loading

def create_clip_model(
    model_spec: str,
    pretrained: Optional[str] = None,
    force_image_size: Optional[int] = None,
    device: str = "auto",
) -> Tuple:
    """
    Load a CLIP model and its preprocessing transform via open_clip.

    Supports three model source types via model_spec:
        - Built-in name:      "ViT-B-32"
        - HuggingFace Hub:    "hf-hub:org/repo"
        - Local directory:    "local-dir:/path/to/model"

    For hf-hub and local-dir sources, pretrained is not passed to open_clip.
    For built-in names, pretrained is passed as the weights tag.

    After loading, transformer blocks are monkey-patched once to enable
    persistent attention storage (model.visual._attn_storage).

    Inputs:
        model_spec        : model identifier string
        pretrained        : pretrained weights tag (ignored for hf-hub/local-dir)
        force_image_size  : if set, override the model's default input resolution
        device            : 'cpu', 'cuda', or 'auto'

    Outputs:
        (model, preprocess, config)
        config = {'patch_size': int, 'image_size': int}
    """
    device = handle_device(device)

    is_hf = model_spec.startswith("hf-hub:")
    is_local = model_spec.startswith("local-dir:")

    kwargs = {}
    if force_image_size is not None:
        kwargs["force_image_size"] = force_image_size

    if is_hf or is_local:
        print(f"Loading model from source: {model_spec}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_spec,
            device=device,
            **kwargs,
        )
    else:
        print(f"Loading model: {model_spec} (pretrained={pretrained})")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_spec,
            pretrained=pretrained,
            device=device,
            **kwargs,
        )

    model.eval()

    patch_size = model.visual.patch_size
    image_size = model.visual.image_size

    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
    if isinstance(image_size, tuple):
        image_size = image_size[0]

    config = {
        "patch_size": int(patch_size),
        "image_size": int(image_size),
    }

    # Attach persistent attention storage via one-time monkey-patch
    model.visual._attn_storage = []
    _monkeypatch_attention_for_open_clip(model)

    return model, preprocess, config


# %% Monkey-patch attention storage

def _monkeypatch_attention_for_open_clip(model) -> None:
    """
    Monkey-patch each ResidualAttentionBlock's attention method once.

    For each block, the original block.attention method is replaced with a
    wrapper that:
        1. Calls the original self.attn module with need_weights=True and
           average_attn_weights=False to capture per-head attention weights.
        2. Appends (block_idx, attn_weights.detach()) to
           model.visual._attn_storage.
        3. Returns only the attention output (not the weights), preserving
           the original method's return contract.

    Stored tensors are shaped (B, H, N, N).

    This function is idempotent with respect to the model's forward behavior.
    Call once after model.eval() in create_clip_model().

    Inputs:
        model : open_clip CLIP model (modified in place)
    """
    transformer = model.visual.transformer
    storage_ref = model.visual  # used as stable reference inside closures

    for block_idx, block in enumerate(transformer.resblocks):
        original_attn = block.attn  # nn.MultiheadAttention

        def make_patched_attention(idx, attn_module):
            def patched_attention(q_x, k_x=None, v_x=None, attn_mask=None, **kwargs):
                k_x = q_x if k_x is None else k_x
                v_x = q_x if v_x is None else v_x

                with torch.no_grad() if not q_x.requires_grad else torch.enable_grad():
                    attn_out, attn_weights = attn_module(
                        q_x, k_x, v_x,
                        need_weights=True,
                        average_attn_weights=False,
                        attn_mask=attn_mask,
                    )

                storage_ref._attn_storage.append((idx, attn_weights.detach().cpu()))
                return attn_out
            return patched_attention

        block.attention = make_patched_attention(block_idx, original_attn)


def _clear_attention_storage(model) -> None:
    """
    Clear all accumulated attention tensors before a new forward pass.

    Inputs:
        model : open_clip CLIP model with _attn_storage attached
    """
    model.visual._attn_storage = []


def _get_attention_storage(model) -> List[Tuple[int, torch.Tensor]]:
    """
    Return the list of (block_idx, attn_tensor) tuples captured during
    the most recent forward pass.

    Inputs:
        model : open_clip CLIP model with _attn_storage attached

    Outputs:
        List of (block_idx, attn_tensor) where attn_tensor is (B, H, N, N)
    """
    return list(model.visual._attn_storage)


# %% Head reduction

def _reduce_heads(attn: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    Reduce multi-head attention along the head dimension.

    Inputs:
        attn      : (B, H, N, N) attention weights
        reduction : 'mean' or 'max'

    Outputs:
        (B, N, N) reduced attention
    """
    if reduction == "mean":
        return attn.mean(dim=1)
    elif reduction == "max":
        return attn.max(dim=1).values
    else:
        raise ValueError(f"head_reduction must be 'mean' or 'max', got '{reduction}'")


# %% Relative-depth layer resolution

def _resolve_layer_index(layer: float, n_layers: int) -> int:
    """
    Map a relative-depth layer value in [0, 1] to the nearest valid block index.

    Inputs:
        layer    : relative depth, where 0.0 = first block, 1.0 = last block
        n_layers : total number of transformer blocks

    Outputs:
        Integer layer index in [0, n_layers - 1]
    """
    idx = round(layer * (n_layers - 1))
    return max(0, min(idx, n_layers - 1))


# %% Attention map helper

def compute_attention_map_from_storage(
    attn_storage: List[Tuple[int, torch.Tensor]],
    layer: float,
    head_reduction: str,
    query_token_index: int = 0,
) -> np.ndarray:
    """
    Derive an attention map from raw storage captured during a forward pass.

    Selects the requested layer by relative depth, reduces heads, and extracts
    the CLS-to-patch spatial vector. No final sum normalization is applied.

    Inputs:
        attn_storage      : list of (block_idx, attn_tensor), attn shape (B, H, N, N)
        layer             : relative depth in [0, 1] (0.0 = first, 1.0 = last)
        head_reduction    : 'mean' or 'max'
        query_token_index : which token row to use as query (0 = CLS)

    Outputs:
        (B, n_patches) float32 ndarray — raw attention weights, not sum-normalized
    """
    if not attn_storage:
        raise ValueError("attn_storage is empty; no attention tensors were captured.")

    sorted_attn = sorted(attn_storage, key=lambda x: x[0])
    n_layers = len(sorted_attn)
    layer_idx = _resolve_layer_index(layer, n_layers)

    _, attn = sorted_attn[layer_idx]              # (B, H, N, N)
    attn = _reduce_heads(attn, head_reduction)    # (B, N, N)
    spatial = attn[:, query_token_index, 1:]      # (B, n_patches); drop CLS column
    return spatial.cpu().float().numpy()


# %% Attention rollout helper

def compute_attention_rollout_from_storage(
    attn_storage: List[Tuple[int, torch.Tensor]],
    head_reduction: str = "max",
    discard_ratio: float = 0.95,
    query_token_index: int = 0,
) -> np.ndarray:
    """
    Compute attention rollout across all transformer layers from raw storage.

    Applies Abnar & Zuidema (2020) rollout with head reduction, residual
    addition, and per-layer row normalization. No final sum normalization is
    applied to the output spatial vector.

    Inputs:
        attn_storage      : list of (block_idx, attn_tensor), attn shape (B, H, N, N)
        head_reduction    : 'mean' or 'max'
        discard_ratio     : fraction of lowest attention values zeroed out per layer
        query_token_index : which token row to read from the final rollout (0 = CLS)

    Outputs:
        (B, n_patches) float32 ndarray — rollout weights, not sum-normalized
    """
    if not attn_storage:
        raise ValueError("attn_storage is empty; no attention tensors were captured.")

    sorted_attn = sorted(attn_storage, key=lambda x: x[0])
    B = sorted_attn[0][1].shape[0]
    N = sorted_attn[0][1].shape[-1]   # 1 (CLS) + n_patches

    rollout = (
        torch.eye(N, device=sorted_attn[0][1].device)
        .unsqueeze(0)
        .expand(B, -1, -1)
        .clone()
    )

    for _, attn in sorted_attn:
        attn_reduced = _reduce_heads(attn, head_reduction)  # (B, N, N)

        # Zero out the lowest discard_ratio fraction of attention entries
        flat = attn_reduced.view(B, -1)
        num_discard = int(flat.size(-1) * discard_ratio)
        if num_discard > 0:
            _, indices = flat.topk(num_discard, dim=-1, largest=False)
            mask = torch.ones_like(flat, dtype=torch.bool)
            mask.scatter_(1, indices, False)
            flat = flat * mask.to(flat.dtype)
        attn_reduced = flat.view(B, N, N)

        # Add residual connection and re-normalise rows
        attn_reduced = attn_reduced + torch.eye(N, device=attn_reduced.device).unsqueeze(0)
        row_sum = attn_reduced.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn_reduced = attn_reduced / row_sum

        rollout = torch.bmm(attn_reduced, rollout)

    spatial = rollout[:, query_token_index, 1:]   # (B, n_patches); drop CLS column
    return spatial.cpu().float().numpy()


# %% Main extraction function

def extract_clip(
    image_batch: List,
    model,
    preprocess,
    device: str = "auto",
    need_attention: bool = False,
) -> dict:
    """
    Run one CLIP image encoding forward pass and return raw base outputs.

    If need_attention is True, attention storage is cleared before the forward
    pass and the captured tensors are returned for downstream metric computation.
    All metric-specific configuration (layer, head_reduction, etc.) is applied
    by the caller via compute_attention_map_from_storage() and
    compute_attention_rollout_from_storage().

    Inputs:
        image_batch    : list of PIL Images or (H, W, 3) uint8 ndarrays
        model          : open_clip model (eval mode, monkey-patched)
        preprocess     : open_clip preprocessing transform
        device         : 'cpu', 'cuda', or 'auto'
        need_attention : if True, clear storage before the forward and return
                         captured attn_storage after

    Outputs:
        dict with keys:
            'embeddings'   : (B, D) float32 ndarray (raw, not normalized)
            'attn_storage' : list of (block_idx, attn_tensor) if need_attention,
                             else empty list
    """
    from PIL import Image as PILImage

    device = handle_device(device)

    # Preprocess images into a batch tensor
    tensors = []
    for img in image_batch:
        if isinstance(img, np.ndarray):
            img = PILImage.fromarray(img)
        tensors.append(preprocess(img))
    batch_tensor = torch.stack(tensors).to(device)

    # Clear storage before the pass if attention will be read afterward
    #if need_attention:
    _clear_attention_storage(model)

    with torch.no_grad():
        embeddings = model.encode_image(batch_tensor)

    attn_storage = _get_attention_storage(model) if need_attention else []

    return {
        "embeddings": embeddings.cpu().float().numpy(),
        "attn_storage": attn_storage,
    }


# %% Embedding-only convenience function

def extract_embeddings(
    image_batch: List,
    model,
    preprocess,
    device: str = "auto",
) -> np.ndarray:
    """
    Extract raw CLIP embeddings without capturing attention.

    Inputs:
        image_batch : list of PIL Images or (H, W, 3) uint8 ndarrays
        model       : open_clip model
        preprocess  : open_clip preprocessing transform
        device      : 'cpu', 'cuda', or 'auto'

    Outputs:
        (B, D) float32 ndarray (not normalized)
    """
    result = extract_clip(image_batch, model, preprocess, device=device, need_attention=False)
    return result["embeddings"]
