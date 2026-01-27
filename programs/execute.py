import argparse
import sys
from pathlib import Path

# ============================================================
# Ensure programs/ is on PYTHONPATH
# (so execute.py can import step1/2/3 safely)
# ============================================================
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# ============================================================
# Import step modules
# ============================================================
from step1_core_extraction import extract_cores
from step2_segmentation import segment_run
from step3_quantification import run_step3


# ============================================================
# Utilities
# ============================================================
def resolve_in_input(filename: str, input_dir: str = "input") -> Path:
    """
    Allow either full path or ./input/<filename>
    """
    p = Path(filename)
    if p.exists():
        return p
    p2 = Path(input_dir) / filename
    if p2.exists():
        return p2
    raise FileNotFoundError(f"File not found: '{filename}' (also tried '{p2}')")


def banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"▶ {title}")
    print("=" * 80)


# ============================================================
# Main pipeline
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description=(
            "CODEX full pipeline\n"
            "  Step1: WSI -> core OME-TIFFs\n"
            "  Step2: core OME-TIFFs -> segmentation masks\n"
            "  Step3: core + mask -> per-cell CSV"
        )
    )

    # ---- user-facing 5 parameters only ----
    ap.add_argument("--image", required=True, help="WSI filename (searched under ./input/ by default)")
    ap.add_argument("--spot", required=True, help="Spot table filename (searched under ./input/ by default)")
    ap.add_argument("--marker", required=True, help="markers.csv (searched under ./input/ by default)")
    ap.add_argument("--channel", required=True, help='Membrane markers, e.g. "PanCK,CD45,Na/K ATPase"')
    ap.add_argument("--output", required=True, help="Run name / project name")

    args = ap.parse_args()

    # ============================================================
    # Resolve inputs once
    # ============================================================
    try:
        image_path = resolve_in_input(args.image)
        spot_path = resolve_in_input(args.spot)
        marker_csv = resolve_in_input(args.marker)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    run_name = args.output
    membrane_markers = [c.strip() for c in args.channel.split(",") if c.strip()]

    if not membrane_markers:
        print("[ERROR] --channel must contain at least one marker", file=sys.stderr)
        sys.exit(1)

    # ============================================================
    # STEP 1: Core extraction
    # ============================================================
    banner("STEP 1 / 3  Core extraction (WSI -> core OME-TIFFs)")
    try:
        written_cores = extract_cores(
            wsi_path=image_path,
            spot_path=spot_path,
            marker_csv=marker_csv,
            output_run=run_name,
        )
        print(f"✓ Step1 complete. Cores written: {len(written_cores)}")
    except Exception as e:
        print(f"[FATAL] Step1 failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ============================================================
    # STEP 2: Segmentation
    # ============================================================
    banner("STEP 2 / 3  Segmentation (core OME-TIFFs -> mask labels)")
    try:
        written_masks = segment_run(
            ometiff_run=run_name,
            marker_csv=marker_csv,
            membrane_markers=membrane_markers,
        )
        print(f"✓ Step2 complete. Masks written: {len(written_masks)}")
    except Exception as e:
        print(f"[FATAL] Step2 failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ============================================================
    # STEP 3: Quantification
    # ============================================================
    banner("STEP 3 / 3  Quantification (core + mask -> per-cell CSV)")
    try:
        out_csv = run_step3(
            run_name=run_name,
            marker_csv=marker_csv,
        )
        print("✓ Step3 complete.")
        print(f"✓ Final output CSV: {out_csv}")
    except Exception as e:
        print(f"[FATAL] Step3 failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ============================================================
    # DONE
    # ============================================================
    banner("PIPELINE FINISHED SUCCESSFULLY")
    print(f"Run name : {run_name}")
    print(f"CSV      : output/step3_output_data/{run_name}/{run_name}.csv")


if __name__ == "__main__":
    main()

# python programs/execute.py --image=20250107RE-TMA-A_Scan1.qptiff --spot=reTMA-A_test3Bspot.txt --marker=markers.csv --channel="PanCK,CD45,Na/K ATPase" --output=test
