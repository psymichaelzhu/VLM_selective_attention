# %%
import yaml
from pathlib import Path
# %%

def create_subset_yml(dataset_name: str) -> None:
    data_dir = "/Users/rezek_zhu/VLM_Mar18/data"
    image_dir = Path(data_dir) / dataset_name / "Image"
    out_path = Path(data_dir) / dataset_name / "subset.yml"

    image_ids = sorted(
        p.name for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    subset = {"all": image_ids}

    with open(out_path, "w") as f:
        yaml.safe_dump(subset, f, sort_keys=False)

    print(f"Saved {len(image_ids)} items to {out_path}")

# %%
create_subset_yml("Hoef2024BRM")

# %%

from shutil import copy2
import os

def copy_images_from_raw_to_Image(dataset_name: str) -> None:
    """
    Copy all images from raw/ (recursively) to Image/ in the same dataset folder.
    Files keep their original filenames.
    """
    data_dir = "/Users/rezek_zhu/VLM_Mar18/data"
    raw_dir = Path(data_dir) / dataset_name / "raw"
    image_dir = Path(data_dir) / dataset_name / "Image"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Gather all images (recursively)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]
    )

    for img_path in images:
        dest = image_dir / img_path.name
        print(f"Copying {img_path} -> {dest}")
        copy2(str(img_path), str(dest))
    print(f"Copied {len(images)} images from {raw_dir} to {image_dir}")

# Example usage:
copy_images_from_raw_to_Image("EmoMadrid")
# %%

from pathlib import Path
from PIL import Image
import numpy as np


def crop_images_to_nonwhite_square_and_resize(dataset_name: str) -> None:
    """
    For each image in raw/, find the non-white region, crop to a square
    containing that region, resize to 500x500, and save into Image/.

    Rules:
    1. Crop the maximal square containing the non-white object region
    2. Resize to 500 x 500
    3. If RGBA, fill background with white, then convert to RGB
    """
    data_dir = "/Users/rezek_zhu/VLM_Mar18/data"
    raw_dir = Path(data_dir) / dataset_name / "raw"
    image_dir = Path(data_dir) / dataset_name / "Image"
    image_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    images = sorted(
        [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]
    )

    for img_path in images:
        print(f"Processing {img_path}")

        img = Image.open(img_path)

        # If image has alpha, composite onto white background first
        if img.mode == "RGBA":
            white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(white_bg, img).convert("RGB")
        else:
            img = img.convert("RGB")

        arr = np.array(img)  # H x W x 3

        # Non-white mask:
        # pixel is considered non-white if any channel is < 250
        # You can tighten or loosen this threshold if needed
        nonwhite_mask = np.any(arr < 250, axis=2)

        ys, xs = np.where(nonwhite_mask)

        if len(xs) == 0 or len(ys) == 0:
            print(f"Skipping {img_path}, no non-white region found")
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        box_w = x_max - x_min + 1
        box_h = y_max - y_min + 1
        side = max(box_w, box_h)

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        img_w, img_h = img.size

        # Initial square
        left = int(round(cx - side / 2))
        top = int(round(cy - side / 2))
        right = left + side
        bottom = top + side

        # Shift square back into image bounds while keeping side length if possible
        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > img_w:
            left -= (right - img_w)
            right = img_w
        if bottom > img_h:
            top -= (bottom - img_h)
            bottom = img_h

        left = max(0, left)
        top = max(0, top)
        right = min(img_w, right)
        bottom = min(img_h, bottom)

        # Final safety: ensure square if touching boundary reduced one side
        crop_w = right - left
        crop_h = bottom - top
        final_side = min(crop_w, crop_h)

        # Re-center final square within current crop
        left = left + (crop_w - final_side) // 2
        top = top + (crop_h - final_side) // 2
        right = left + final_side
        bottom = top + final_side

        cropped = img.crop((left, top, right, bottom))
        resized = cropped.resize((500, 500), Image.Resampling.LANCZOS)

        out_path = image_dir / img_path.name
        resized.save(out_path)

    print(f"Processed {len(images)} images from {raw_dir} to {image_dir}")
# %%
crop_images_to_nonwhite_square_and_resize("Hoef2024BRM")
# %%

from pathlib import Path
from PIL import Image, ImageChops


def has_black_edge(img: Image.Image) -> bool:
    """Return True if any edge (top/bottom/left/right) is a solid black strip."""
    arr = img.load()
    w, h = img.size

    def is_black(pixel):
        return all(c <= 10 for c in pixel[:3])

    return (
        all(is_black(arr[x, 0])   for x in range(w)) or  # top
        all(is_black(arr[x, h-1]) for x in range(w)) or  # bottom
        all(is_black(arr[0, y])   for y in range(h)) or  # left
        all(is_black(arr[w-1, y]) for y in range(h))     # right
    )

def filter_and_resize_no_black_edge(dataset_name="EmoMadrid",
                                   base_dir="/Users/rezek_zhu/VLM_Mar18/data",
                                   strip_white_border=False):
    raw_dir = Path(base_dir) / dataset_name / "raw"
    image_dir = Path(base_dir) / dataset_name / "Image"
    black_edge_dir = Path(base_dir) / dataset_name / "black_edge"
    image_dir.mkdir(parents=True, exist_ok=True)
    black_edge_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(raw_dir.rglob("*")):
        if img_path.is_file() and img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            img = Image.open(img_path).convert("RGB")
        else:
            continue
        if strip_white_border:
            # strip white border
            bbox = ImageChops.difference(img, Image.new("RGB", img.size, (255, 255, 255))).getbbox()
            if bbox:
                img = img.crop(bbox)

        if has_black_edge(img):
            img.save(black_edge_dir / img_path.name)
            print(f"black_edge: {img_path.name}")
        else:
            img.resize((500, 500), Image.Resampling.LANCZOS).save(image_dir / img_path.name)
            print(f"Image:      {img_path.name}")
filter_and_resize_no_black_edge("EmoMadrid")

# %%
from pathlib import Path
from PIL import Image, ImageChops


def has_black_edge(img: Image.Image) -> bool:
    """Return True if any edge (top/bottom/left/right) is a solid black strip."""
    arr = img.load()
    w, h = img.size

    def is_black(pixel):
        return all(c <= 10 for c in pixel[:3])

    return (
        all(is_black(arr[x, 0])   for x in range(w)) or  # top
        all(is_black(arr[x, h-1]) for x in range(w)) or  # bottom
        all(is_black(arr[0, y])   for y in range(h)) or  # left
        all(is_black(arr[w-1, y]) for y in range(h))     # right
    )


raw_dir       = Path("raw")
image_dir     = Path("Image")
black_edge_dir = Path("black_edge")
image_dir.mkdir(exist_ok=True)
black_edge_dir.mkdir(exist_ok=True)

for img_path in sorted(raw_dir.glob("*")):
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        continue

    img = Image.open(img_path).convert("RGB")

    # strip white border
    bbox = ImageChops.difference(img, Image.new("RGB", img.size, (255, 255, 255))).getbbox()
    if bbox:
        img = img.crop(bbox)

    if has_black_edge(img):
        img.save(black_edge_dir / img_path.name)
        print(f"black_edge: {img_path.name}")
    else:
        img.resize((500, 500), Image.Resampling.LANCZOS).save(image_dir / img_path.name)
        print(f"Image:      {img_path.name}")


def process_black_edge(dataset_name="EmoMadrid"):
    base_dir = "/Users/rezek_zhu/VLM_Mar18/data"
    black_edge_dir = Path(base_dir) / dataset_name / "black_edge"
    image_be_dir   = Path(base_dir) / dataset_name / "Image_be"
    image_be_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(black_edge_dir.rglob("*")):
        if img_path.is_file() and img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            img = Image.open(img_path).convert("RGB")
        else:
            continue
        img = Image.open(img_path).convert("RGB")

        # strip black border
        bbox = ImageChops.difference(img, Image.new("RGB", img.size, (0, 0, 0))).getbbox()
        if bbox:
            img = img.crop(bbox)

        img.resize((500, 500), Image.Resampling.LANCZOS).save(image_be_dir / img_path.name)
        print(f"Image_be: {img_path.name}")


process_black_edge("EmoMadrid")
# %%
from pathlib import Path
from PIL import Image

def get_black_edge_bbox(img: Image.Image):
    """Return bbox that removes solid black edges using has_black_edge logic."""
    arr = img.load()
    w, h = img.size

    def is_black(pixel):
        return all(c <= 10 for c in pixel[:3])

    # find top
    top = 0
    for y in range(h):
        if all(is_black(arr[x, y]) for x in range(w)):
            top += 1
        else:
            break

    # find bottom
    bottom = h
    for y in range(h - 1, -1, -1):
        if all(is_black(arr[x, y]) for x in range(w)):
            bottom -= 1
        else:
            break

    # find left
    left = 0
    for x in range(w):
        if all(is_black(arr[x, y]) for y in range(h)):
            left += 1
        else:
            break

    # find right
    right = w
    for x in range(w - 1, -1, -1):
        if all(is_black(arr[x, y]) for y in range(h)):
            right -= 1
        else:
            break

    # avoid empty crop
    if left >= right or top >= bottom:
        return None

    return (left, top, right, bottom)


def process_black_edge(dataset_name="EmoMadrid"):
    base_dir = "/Users/rezek_zhu/VLM_Mar18/data"
    black_edge_dir = Path(base_dir) / dataset_name / "black_edge"
    image_be_dir   = Path(base_dir) / dataset_name / "Image_be"
    image_be_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for img_path in sorted(black_edge_dir.rglob("*")):
        if not (img_path.is_file() and img_path.suffix.lower() in valid_exts):
            continue

        img = Image.open(img_path).convert("RGB")

        bbox = get_black_edge_bbox(img)

        if bbox is not None:
            img = img.crop(bbox)

        img.resize((500, 500), Image.Resampling.LANCZOS)\
           .save(image_be_dir / img_path.name)

        print(f"Image_be: {img_path.name}")


process_black_edge("EmoMadrid")
# %%
