import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from deepcell.applications import Mesmer


# =========================
# Small utilities
# =========================

def resolve_in_input(filename: str, input_dir: str = "input") -> Path:
    """Allow user to pass either a full path or a filename under ./input/."""
    p = Path(filename)
    if p.exists():
        return p
    p2 = Path(input_dir) / filename
    if p2.exists():
        return p2
    raise FileNotFoundError(f"File not found: '{filename}' (also tried '{p2}')")


def load_marker_mapping(marker_csv: Path) -> pd.DataFrame:
    """Load markers.csv; expects columns: channel_name (1-based), marker_name."""
    df = pd.read_csv(marker_csv)
    df.columns = [c.strip() for c in df.columns]
    if "channel_name" not in df.columns or "marker_name" not in df.columns:
        raise ValueError("markers.csv must contain 'channel_name' and 'marker_name'")

    df["channel_name"] = pd.to_numeric(df["channel_name"], errors="coerce")
    df = df.dropna(subset=["channel_name"])
    df["channel_name"] = df["channel_name"].astype(int)
    df["marker_name"] = df["marker_name"].astype(str).str.strip()
    return df


def get_channel_indices(marker_df: pd.DataFrame, names: list[str]) -> list[int]:
    """Convert marker names -> 0-based channel indices using mapping from markers.csv."""
    mapping = dict(zip(marker_df["marker_name"], marker_df["channel_name"] - 1))
    idx = []
    for n in names:
        n2 = str(n).strip()
        if n2 in mapping:
            idx.append(int(mapping[n2]))
    return idx


def parse_channels_csv_arg(channel_csv: str) -> list[str]:
    """
      --channel "PanCK,CD45,Na/K ATPase"
    """
    if channel_csv is None:
        return []
    parts = [p.strip() for p in str(channel_csv).split(",")]
    return [p for p in parts if p]


def read_ome_as_batch(core_path: Path) -> np.ndarray:
    """
    Read an OME-TIFF core into DeepCell batch shape: (1, H, W, C).
    Step1 outputs axes='CYX' -> tifffile.imread usually returns (C, Y, X),
    so we transpose to (Y, X, C).
    """
    img = tifffile.imread(str(core_path))

    if img.ndim == 3:
        # If CYX: channels dimension is small (e.g. 37) compared to X
        if img.shape[0] < img.shape[2]:
            img = np.transpose(img, (1, 2, 0))  # (Y,X,C)
        batch = np.expand_dims(img, 0)          # (1,Y,X,C)
        return batch

    if img.ndim == 4:
        # Try to normalize to (N,Y,X,C)
        if img.shape[-1] < img.shape[1] and img.shape[-1] < img.shape[2]:
            # likely (N,C,Y,X)
            return np.transpose(img, (0, 2, 3, 1))
        return img

    raise ValueError(f"Unexpected OME-TIFF shape: {img.shape}")


def save_mask(mask_path: Path, mask: np.ndarray, mpp: float) -> None:
    """Save label mask as uint32 tif."""
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(mask_path),
        mask.astype("uint32"),
        compression="zlib",
        resolution=(1 / mpp, 1 / mpp),
        metadata={"unit": "um"},
    )


# =========================
# Segmentation core logic
# =========================

NAME_NUCLEAR = "DAPI"   # fixed as you requested
DEFAULT_MPP = 0.5075    # keep same as your colleague's script


def segment_one_core(
    app: Mesmer,
    core_ometif_path: Path,
    marker_df: pd.DataFrame,
    membrane_markers: list[str],
    mpp: float,
) -> np.ndarray:
    """Input: one core OME-TIFF -> Output: 2D label mask (H,W)."""
    batch = read_ome_as_batch(core_ometif_path)  # (1,H,W,C)

    idx_nuc_list = get_channel_indices(marker_df, [NAME_NUCLEAR])
    if not idx_nuc_list:
        raise ValueError(f"Required nuclear channel '{NAME_NUCLEAR}' not found in markers.csv")
    idx_nuc = idx_nuc_list[0]

    idx_mem = get_channel_indices(marker_df, membrane_markers)
    if not idx_mem:
        raise ValueError(f"None of membrane markers found in markers.csv: {membrane_markers}")

    nuc = batch[..., idx_nuc: idx_nuc + 1]  # (1,H,W,1)
    mem = np.mean(batch[..., idx_mem], axis=-1, keepdims=True)  # (1,H,W,1)
    inp = np.concatenate((nuc, mem), axis=-1)  # (1,H,W,2)

    pred = app.predict(inp, image_mpp=mpp, compartment="whole-cell")
    mask = np.squeeze(pred)  # (H,W)
    return mask


def segment_run(
    ometiff_run: str,
    marker_csv: Path,
    membrane_markers: list[str],
    mpp: float = DEFAULT_MPP,
) -> list[Path]:
    """
    Segment all cores under:
      output/step1_output_data/<ometiff_run>/*.ome.tif

    Output masks to:
      output/step2_output_data/<ometiff_run>/*_SEG.tif
    """
    in_dir = Path("output") / "step1_output_data" / ometiff_run
    if not in_dir.exists():
        raise FileNotFoundError(f"Input run folder not found: {in_dir}")

    out_dir = Path("output") / "step2_output_data" / ometiff_run
    out_dir.mkdir(parents=True, exist_ok=True)

    marker_df = load_marker_mapping(marker_csv)

    core_files = sorted(list(in_dir.glob("*.tif")) + list(in_dir.glob("*.tiff")))
    if not core_files:
        raise FileNotFoundError(f"No .tif/.tiff found in {in_dir}")

    app = Mesmer()  # init once

    written = []
    for core_path in tqdm(core_files, desc="Segmenting cores"):
        base = core_path.stem  # e.g., "H-1.ome" for "H-1.ome.tif"
        if base.endswith(".ome"):
            base = base[:-4]  # -> "H-1"

        mask_path = out_dir / f"{base}_SEG.tif"

        try:
            mask = segment_one_core(
                app=app,
                core_ometif_path=core_path,
                marker_df=marker_df,
                membrane_markers=membrane_markers,
                mpp=mpp,
            )
            save_mask(mask_path, mask, mpp)
            written.append(mask_path)
        finally:
            gc.collect()

    return written


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="Step2: segmentation (core OME-TIFFs -> mask label TIFs)")
    ap.add_argument(
        "--ometiff",
        required=True,
        help="Run name from step1 (e.g., test). Reads output/step1_output_data/<run>/",
    )
    ap.add_argument(
        "--marker",
        required=True,
        help="markers.csv filename (default searched under ./input/)",
    )
    ap.add_argument(
        "--channel",
        required=True,
        help='Comma-separated membrane markers, e.g. "PanCK,CD45,Na/K ATPase"',
    )
    args = ap.parse_args()

    marker_csv = resolve_in_input(args.marker)
    membrane_markers = parse_channels_csv_arg(args.channel)

    written = segment_run(
        ometiff_run=args.ometiff,
        marker_csv=marker_csv,
        membrane_markers=membrane_markers,
    )

    print(f"\n✅ Done. Wrote {len(written)} masks to output/step2_output_data/{args.ometiff}/")
    if written:
        print(f"   Example: {written[0]}")


if __name__ == "__main__":
    main()
# H-1.ome.tif  → 原始強度影像（37 channels）       H-1_SEG.tif  → 分割結果（label mask）
# python programs/step2_segmentation.py --ometiff=test --marker=markers.csv --channel="PanCK,CD45,Na/K ATPase"
