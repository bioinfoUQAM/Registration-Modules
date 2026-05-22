"""
apply_registration_to_bboxes.py
---------------------------------
Apply the best 7-DOF affine transform (found by stall_affine_search.py) to
all normalised bounding box columns in a scaled CSV.

Requires stall_affine_search.py in the same directory.

Usage:
    python apply_registration_to_bboxes.py \
        --input  path/to/animal_scaled.csv \
        --output path/to/animal_scaled_reg.csv \
        --floating  x1 y1 x2 y2 x3 y3 x4 y4 \
        [--standard x1 y1 x2 y2 x3 y3 x4 y4] \
        [--nr 7] \
        [--prefixes "body box" "head box" "snout box"]
"""

import argparse
import os
import numpy as np
import pandas as pd
import stall_affine_search


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def affine_coeffs_from_params(tx, ty, rot_c, rot_s, sx, sy, skx, sky):
    a00 = sx * (rot_c + sky * rot_s);  a01 = sx * (skx * rot_c + rot_s);  a02 = tx
    a10 = sy * (-rot_s + sky * rot_c); a11 = sy * (-skx * rot_s + rot_c); a12 = ty
    return a00, a01, a02, a10, a11, a12


def apply_affine(points_xy, a00, a01, a02, a10, a11, a12):
    x = points_xy[:, 0];  y = points_xy[:, 1]
    return np.stack([x * a00 + y * a01 + a02,
                     x * a10 + y * a11 + a12], axis=1)


def bbox_to_corners(Ln, Tn, Wn, Hn):
    """Convert normalised TLWH bbox to 4 corners (LT, LB, RB, RT)."""
    x1, y1 = Ln, Tn
    x2, y2 = Ln + 2.0 * Wn, Tn + 2.0 * Hn
    return np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.float64)


def corners_to_bbox(corners_4x2):
    """Convert 4 corners back to normalised TLWH bbox."""
    mn = corners_4x2.min(axis=0);  mx = corners_4x2.max(axis=0)
    Ln, Tn = mn
    Wn = 0.5 * (mx[0] - mn[0])
    Hn = 0.5 * (mx[1] - mn[1])
    return Ln, Tn, Wn, Hn


def read_table_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":   return pd.read_csv(path)
    if ext in [".xlsx", ".xls"]: return pd.read_excel(path)
    raise ValueError(f"Unsupported format: {path}")


# ---------------------------------------------------------------------------
# Core registration function
# ---------------------------------------------------------------------------

def register_bboxes(df, affine_coeffs, prefixes):
    """
    Apply a 7-DOF affine transform to normalised bounding box columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature table with columns <prefix> Ln/Tn/Wn/Hn.
    affine_coeffs : tuple (a00, a01, a02, a10, a11, a12)
        Affine matrix coefficients from affine_coeffs_from_params().
    prefixes : list of str
        Body-part column prefixes, e.g. ["body box", "head box", "snout box"].

    Returns
    -------
    pd.DataFrame with added *_reg columns per prefix.
    """
    a00, a01, a02, a10, a11, a12 = affine_coeffs

    for p in prefixes:
        Ln_col, Tn_col = f"{p} Ln", f"{p} Tn"
        Wn_col, Hn_col = f"{p} Wn", f"{p} Hn"
        missing = [c for c in [Ln_col, Tn_col, Wn_col, Hn_col] if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns for '{p}': {missing}")

        Ln = df[Ln_col].to_numpy(dtype=np.float64)
        Tn = df[Tn_col].to_numpy(dtype=np.float64)
        Wn = df[Wn_col].to_numpy(dtype=np.float64)
        Hn = df[Hn_col].to_numpy(dtype=np.float64)

        Ln2 = np.empty_like(Ln);  Tn2 = np.empty_like(Tn)
        Wn2 = np.empty_like(Wn);  Hn2 = np.empty_like(Hn)

        for i in range(len(df)):
            corners   = bbox_to_corners(Ln[i], Tn[i], Wn[i], Hn[i])
            corners_t = apply_affine(corners, a00, a01, a02, a10, a11, a12)
            Ln2[i], Tn2[i], Wn2[i], Hn2[i] = corners_to_bbox(corners_t)

        df[f"{p} Ln_reg"] = Ln2;  df[f"{p} Tn_reg"] = Tn2
        df[f"{p} Wn_reg"] = Wn2;  df[f"{p} Hn_reg"] = Hn2
        df[f"{p} AR_reg"]    = Wn2 / (Hn2 + 1e-9)
        df[f"{p} logAR_reg"] = np.log(df[f"{p} AR_reg"] + 1e-12)
        print(f"  Registered: {p}")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply 7-DOF affine registration to normalised bounding boxes.")
    parser.add_argument("--input",    required=True,
                        help="Input scaled CSV (output of select_check_points.py).")
    parser.add_argument("--output",   required=True,
                        help="Output registered CSV path.")
    parser.add_argument("--floating", type=float, nargs=8, required=True,
                        metavar=("x1","y1","x2","y2","x3","y3","x4","y4"),
                        help="Normalised floating stall corners: LT LB RB RT.")
    parser.add_argument("--standard", type=float, nargs=8,
                        default=[-0.8,-1.0, -0.8,1.0, 0.8,1.0, 0.8,-1.0],
                        metavar=("x1","y1","x2","y2","x3","y3","x4","y4"),
                        help="Canonical stall corners (default: ratio 4:5).")
    parser.add_argument("--nr",       type=int, default=7,
                        help="Grid resolution for affine search (default: 7).")
    parser.add_argument("--prefixes", nargs="+",
                        default=["body box", "head box", "snout box"],
                        help="Body-part column prefixes to register.")
    args = parser.parse_args()

    floating  = np.array(list(zip(args.floating[0::2], args.floating[1::2])), dtype=np.float64)
    standard  = np.array(list(zip(args.standard[0::2], args.standard[1::2])), dtype=np.float64)

    # Step 1 — find best affine transform
    _, (best_idx, best_val) = stall_affine_search.find_local_minima(
        floating, standard, Nr=args.nr)

    tx  = stall_affine_search.tx_vals[best_idx[0]]
    ty  = stall_affine_search.ty_vals[best_idx[1]]
    rot = stall_affine_search.rot_vals[best_idx[2]]
    sx  = stall_affine_search.sx_vals[best_idx[3]]
    sy  = stall_affine_search.sy_vals[best_idx[4]]
    skx = stall_affine_search.skx_vals[best_idx[5]]
    sky = stall_affine_search.sky_vals[best_idx[6]]

    print(f"\nBest transform: value={best_val:.6f}")
    print(f"  tx={tx:.4f} ty={ty:.4f} rot={rot:.4f} "
          f"sx={sx:.4f} sy={sy:.4f} skx={skx:.4f} sky={sky:.4f}")

    rot_c, rot_s = np.cos(rot), np.sin(rot)
    coeffs = affine_coeffs_from_params(tx, ty, rot_c, rot_s, sx, sy, skx, sky)

    # Step 2 — load table and apply
    df = read_table_any(args.input)
    print(f"Loaded: {args.input}  ({df.shape[0]} rows)")

    df = register_bboxes(df, coeffs, args.prefixes)

    # Step 3 — save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
