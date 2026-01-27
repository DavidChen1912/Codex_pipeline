import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import openslide
from tqdm import tqdm


def resolve_in_input(filename: str, input_dir: str = "input") -> Path:
    p = Path(filename)
    if p.exists():
        return p
    p2 = Path(input_dir) / filename
    if p2.exists():
        return p2
    raise FileNotFoundError(f"File not found: '{filename}' (also tried '{p2}')")


def sanitize_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name or "unnamed"


def load_marker_names(marker_csv: Path) -> list[str]:
    df = pd.read_csv(marker_csv)
    df.columns = df.columns.str.strip()

    if "channel_name" not in df.columns or "marker_name" not in df.columns:
        raise ValueError("markers.csv must contain 'channel_name' and 'marker_name'")

    df["channel_name"] = pd.to_numeric(df["channel_name"], errors="coerce")
    df = df.dropna(subset=["channel_name"]).sort_values("channel_name")

    return df["marker_name"].astype(str).str.strip().tolist()


def read_slide_props(wsi_path: Path) -> tuple[float, float, float, float]:
    slide = openslide.OpenSlide(str(wsi_path))
    try:
        mpp_x = float(slide.properties.get("openslide.mpp-x", 0.5))
        mpp_y = float(slide.properties.get("openslide.mpp-y", 0.5))
        bounds_x = float(slide.properties.get("openslide.bounds-x", 0))
        bounds_y = float(slide.properties.get("openslide.bounds-y", 0))
    finally:
        slide.close()
    return mpp_x, mpp_y, bounds_x, bounds_y


def detect_channel_pages(wsi_path: Path) -> list[int]:
    real = []
    with tifffile.TiffFile(str(wsi_path)) as tif:
        base_shape = tif.pages[0].shape
        for i, page in enumerate(tif.pages):
            if page.shape == base_shape:
                real.append(i)
            else:
                break
    if not real:
        raise RuntimeError("Could not detect channel pages from WSI.")
    return real


def extract_cores(
    wsi_path: Path,
    spot_path: Path,
    marker_csv: Path,
    output_run: str,
    core_size: int = 4000,
    tile_size: int = 256,
) -> list[Path]:
    """
    主功能：讀 WSI + spot table + markers
    切出每一顆 core，存成一個 OME-TIFF。
    回傳所有寫出的檔案路徑。
    """
    # output/step1_output_data/<run>/
    out_dir = Path("output") / "step1_output_data" / output_run
    out_dir.mkdir(parents=True, exist_ok=True)

    marker_names = load_marker_names(marker_csv)
    spots = pd.read_csv(spot_path, sep="\t")
    spots.columns = spots.columns.str.strip()

    mpp_x, mpp_y, bounds_x, bounds_y = read_slide_props(wsi_path)
    channel_pages = detect_channel_pages(wsi_path)
    num_channels = len(channel_pages)

    # markers 與真實 channel 對齊：多就截斷，少就補 Channel i
    if len(marker_names) > num_channels:
        marker_names = marker_names[:num_channels]
    elif len(marker_names) < num_channels:
        for i in range(len(marker_names), num_channels):
            marker_names.append(f"Channel {i+1}")

    written = []

    with tifffile.TiffFile(str(wsi_path)) as tif:
        base_h, base_w = tif.pages[channel_pages[0]].shape
        dtype = tif.pages[channel_pages[0]].dtype

        for _, row in tqdm(spots.iterrows(), total=len(spots), desc="Extracting cores"):
            if str(row.get("Missing core", "")).lower() == "true":
                continue

            core_name_raw = row["Name"]
            core_name = sanitize_name(core_name_raw)

            # spot 檔給的是 µm 的 centroid，轉成 pixel
            cx_um = float(row["Centroid X µm"])
            cy_um = float(row["Centroid Y µm"])
            cx_px = (cx_um / mpp_x) + bounds_x
            cy_px = (cy_um / mpp_y) + bounds_y

            top_left_x = int(cx_px - core_size / 2)
            top_left_y = int(cy_px - core_size / 2)

            # clamp
            top_left_x = max(0, min(top_left_x, base_w - 1))
            top_left_y = max(0, min(top_left_y, base_h - 1))

            x_end = min(top_left_x + core_size, base_w)
            y_end = min(top_left_y + core_size, base_h)

            h_real = y_end - top_left_y
            w_real = x_end - top_left_x
            if h_real <= 0 or w_real <= 0:
                continue

            img_stack = np.zeros((num_channels, core_size, core_size), dtype=dtype)

            for c_idx, page_idx in enumerate(channel_pages):
                crop = tif.pages[page_idx].asarray(out="memmap")[top_left_y:y_end, top_left_x:x_end]
                img_stack[c_idx, 0:h_real, 0:w_real] = crop

            save_path = out_dir / f"{core_name}.ome.tif"

            metadata = {
                "PhysicalSizeX": mpp_x,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": mpp_y,
                "PhysicalSizeYUnit": "µm",
                "axes": "CYX",
                "Channel": {"Name": marker_names},
                "Name": str(core_name_raw),
            }

            tifffile.imwrite(
                str(save_path),
                img_stack,
                photometric="minisblack",
                compression="lzw",
                tile=(tile_size, tile_size),
                metadata=metadata,
            )

            written.append(save_path)

    return written


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Step1: core extraction (WSI -> core-level OME-TIFFs)")
    ap.add_argument("--image", required=True, help="WSI filename (default searched under ./input/)")
    ap.add_argument("--spot", required=True, help="Spot table filename (default searched under ./input/)")
    ap.add_argument("--marker", required=True, help="markers.csv filename (default searched under ./input/)")
    ap.add_argument("--output", required=True, help="Run name. Output -> output/step1_output_data/<run>/")
    args = ap.parse_args()

    wsi_path = resolve_in_input(args.image)
    spot_path = resolve_in_input(args.spot)
    marker_csv = resolve_in_input(args.marker)

    written = extract_cores(
        wsi_path=wsi_path,
        spot_path=spot_path,
        marker_csv=marker_csv,
        output_run=args.output,
    )

    print(f"\n✅ Done. Wrote {len(written)} core OME-TIFFs to output/step1_output_data/{args.output}/")
    if written:
        print(f"   Example: {written[0]}")


if __name__ == "__main__":
    main()

# python programs/step1_core_extraction.py --image=20250107RE-TMA-A_Scan1.qptiff --spot=reTMA-A_test3Bspot.txt --marker=markers.csv --output=test
