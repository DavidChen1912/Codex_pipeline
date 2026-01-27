# - Input:
#   1) output/step1_output_data/<run>/*.ome.tif   (core OME-TIFF; multi-channel intensity)
#   2) output/step2_output_data/<run>/*_SEG.tif   (mask label image; uint32)
#   3) input/<marker.csv>                         (channel_name -> marker_name)
# - Output:
#   output/step3_output_data/<run>/<run>.csv       (one row per cell)  
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm
from skimage.measure import regionprops_table


# ----------------------------
# Utilities
# ----------------------------
def resolve_in_input(filename: str, input_dir: str = "input") -> Path:
    """
    If `filename` exists as a path, use it.
    Otherwise, look under ./input/<filename>.
    """
    p = Path(filename)
    if p.exists():
        return p
    p2 = Path(input_dir) / filename
    if p2.exists():
        return p2
    raise FileNotFoundError(f"File not found: '{filename}' (also tried '{p2}')")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_marker_mapping(marker_csv: Path) -> pd.DataFrame:
    """
    markers.csv must contain:
      - channel_name : 1-based channel index (int)
      - marker_name  : marker name (str)
    """
    df = pd.read_csv(marker_csv)
    df.columns = [c.strip() for c in df.columns]

    if "channel_name" not in df.columns or "marker_name" not in df.columns:
        raise ValueError("markers.csv must contain 'channel_name' and 'marker_name' columns")

    df["channel_name"] = pd.to_numeric(df["channel_name"], errors="coerce")
    df = df.dropna(subset=["channel_name"]).copy()
    df["channel_name"] = df["channel_name"].astype(int)
    df["marker_name"] = df["marker_name"].astype(str).str.strip()
    return df


def list_core_files(step1_run_dir: Path) -> list[Path]:
    """
    Step1 outputs core images as:
      <core_name>.ome.tif
    This returns all .tif/.tiff files under the run directory.
    """
    files = sorted(step1_run_dir.glob("*.tif")) + sorted(step1_run_dir.glob("*.tiff"))
    # unique + stable order
    files = sorted({f.resolve() for f in files})
    return list(files)


def core_id_from_filename(core_path: Path) -> str:
    """
    Convert core filename -> core_id that matches step2 mask naming.

    Expected step1 naming (recommended):
      H-1.ome.tif   -> core_id = H-1
    Mask naming (step2):
      H-1_SEG.tif

    This function strips ".ome.tif" or ".ome.tiff" if present.
    """
    name = core_path.name
    if name.endswith(".ome.tif"):
        return name[: -len(".ome.tif")]
    if name.endswith(".ome.tiff"):
        return name[: -len(".ome.tiff")]
    # fallback: strip one extension
    return core_path.stem


def read_core_as_hwc(core_path: Path) -> np.ndarray:
    """
    Read core OME-TIFF and return image as (H, W, C).
    Step1 wrote axes='CYX' so tifffile.imread usually returns (C, Y, X).
    We'll convert to (Y, X, C).
    """
    img = tifffile.imread(str(core_path))
    if img.ndim != 3:
        raise ValueError(f"Core image must be 3D. Got shape={img.shape} ({core_path.name})")

    # typical: (C, Y, X) where C ~ 37
    # if already (Y, X, C), last dim is small
    if img.shape[0] < img.shape[-1]:
        img = np.transpose(img, (1, 2, 0))
    return img  # (H, W, C)


def read_mask_hw(mask_path: Path) -> np.ndarray:
    """
    Read mask tif and return as 2D label image (H, W).
    """
    m = tifffile.imread(str(mask_path))
    m = np.squeeze(m)
    if m.ndim != 2:
        raise ValueError(f"Mask must be 2D. Got shape={m.shape} ({mask_path.name})")
    return m


# ----------------------------
# Quantification core
# ----------------------------
def quantify_one_core(
    core_img_hwc: np.ndarray,
    mask_hw: np.ndarray,
    marker_df: pd.DataFrame,
    image_id: str,
) -> pd.DataFrame:
    """
    Produce per-cell table for a single core.
    - Morphology: label, centroid, area, eccentricity, solidity
    - Intensity: mean intensity per marker channel (from markers.csv)
    """
    if core_img_hwc.shape[:2] != mask_hw.shape:
        raise ValueError(f"Core (H,W)={core_img_hwc.shape[:2]} != Mask={mask_hw.shape}")

    # Step A: morphology
    props = ["label", "centroid", "area", "eccentricity", "solidity"]
    base = regionprops_table(mask_hw, properties=props)
    df = pd.DataFrame(base)
    df = df.rename(columns={"centroid-0": "Cell_Y_Pos", "centroid-1": "Cell_X_Pos"})

    # Step B: mean intensity per channel/marker
    num_channels = core_img_hwc.shape[-1]

    for _, row in marker_df.iterrows():
        marker_name = str(row["marker_name"]).strip()
        ch0 = int(row["channel_name"]) - 1  # 1-based -> 0-based
        if 0 <= ch0 < num_channels:
            intensity_img = core_img_hwc[:, :, ch0]
            val = regionprops_table(mask_hw, intensity_image=intensity_img, properties=("mean_intensity",))
            df[marker_name] = val["mean_intensity"]

    # Step C: add ImageId + column order
    df["ImageId"] = image_id
    meta_cols = ["ImageId", "label", "Cell_X_Pos", "Cell_Y_Pos", "area", "eccentricity", "solidity"]
    final_cols = meta_cols + [c for c in df.columns if c not in meta_cols]
    return df[final_cols]


def run_step3(run_name: str, marker_csv: Path) -> Path:
    """
    Batch quantification:
      - iterate all cores in step1 run folder
      - find corresponding masks in step2 run folder
      - concatenate to one CSV: output/step3_output_data/<run>/<run>.csv
    """
    step1_dir = Path("output") / "step1_output_data" / run_name
    step2_dir = Path("output") / "step2_output_data" / run_name
    out_dir = Path("output") / "step3_output_data" / run_name
    ensure_dir(out_dir)

    if not step1_dir.exists():
        raise FileNotFoundError(f"Step1 folder not found: {step1_dir}")
    if not step2_dir.exists():
        raise FileNotFoundError(f"Step2 folder not found: {step2_dir}")

    marker_df = load_marker_mapping(marker_csv)
    core_files = list_core_files(step1_dir)
    if not core_files:
        raise FileNotFoundError(f"No core OME-TIFFs found in: {step1_dir}")

    tables: list[pd.DataFrame] = []
    missing_masks = 0
    failed = 0

    for core_path in tqdm(core_files, desc="Step3 Quantification"):
        core_id = core_id_from_filename(core_path)
        mask_path = step2_dir / f"{core_id}_SEG.tif"

        if not mask_path.exists():
            missing_masks += 1
            continue

        try:
            core_img = read_core_as_hwc(core_path)
            mask = read_mask_hw(mask_path)
            df_core = quantify_one_core(core_img, mask, marker_df, image_id=core_id)
            tables.append(df_core)
        except Exception as e:
            failed += 1
            print(f"[ERROR] core={core_path.name}: {e}", file=sys.stderr)

    if not tables:
        raise RuntimeError(
            f"No cores quantified. missing_masks={missing_masks}, failed={failed}. "
            f"Check: (1) step2 outputs exist, (2) mask naming <core_id>_SEG.tif matches cores."
        )

    df_all = pd.concat(tables, ignore_index=True)
    out_csv = out_dir / f"{run_name}.csv"
    df_all.to_csv(out_csv, index=False)

    print("\n✅ Step3 complete.")
    print(f"   Cores found (step1): {len(core_files)}")
    print(f"   Missing masks (step2): {missing_masks}")
    print(f"   Failed cores: {failed}")
    print(f"   Rows (cells): {len(df_all)}")
    print(f"   Output CSV: {out_csv}")
    return out_csv


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Step3: quantification (core OME-TIFF + mask -> per-cell CSV)")
    ap.add_argument("--run", required=True, help="Run name (e.g., test)")
    ap.add_argument("--marker", required=True, help="markers.csv filename (searched under ./input/ by default)")
    args = ap.parse_args()

    marker_csv = resolve_in_input(args.marker)
    run_step3(run_name=args.run, marker_csv=marker_csv)


if __name__ == "__main__":
    main()

# python programs/step3_quantification.py --run=test --marker=markers.csv