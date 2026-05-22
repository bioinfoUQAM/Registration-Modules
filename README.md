# Registration Modules

Two modules for stall registration and bounding box alignment. Part of the [WELL-E](https://github.com/bioinfoUQAM) animal welfare computer vision pipeline.

---

## Modules Overview

| Module | Role |
|--------|------|
| `stall_affine_search.py` | Search for the best 7-DOF affine transform aligning stall corners to a standard reference |
| `apply_registration_to_bboxes.py` | Apply the found transform to all bounding box coordinates |

---

## Module 1 — `stall_affine_search.py`

Searches a 7-dimensional parameter space (translation `tx`, `ty`; rotation; scale `sx`, `sy`; shear `skx`, `sky`) to find the affine transformation that best maps a floating stall (video-specific, normalised) to a standard stall reference.

Uses **Numba JIT + parallel** kernels for the distance grid computation and local minima detection.

### Key function

```python
minima, (best_idx, best_val) = find_local_minima(floating, standard_stall, Nr=7)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `floating` | `(N, 2)` array | Normalised stall corners from the video (LT, LB, RB, RT) |
| `standard_stall` | `(N, 2)` array | Target canonical stall corners |
| `Nr` | `int` | Grid resolution per dimension (default: 7). Total cells = `Nr^7` |

**Returns:** list of local minima + best global minimum index and value.

### Parameter search ranges

| Param | Range |
|-------|-------|
| `tx`, `ty` | `[-0.5, 0.5]` |
| `rot` | `[-0.5, 0.5]` rad |
| `sx`, `sy` | `[0.5, 1.5]` |
| `skx`, `sky` | `[-0.5, 0.5]` |

### Requirements

```bash
pip install numpy numba
```

> First call triggers JIT compilation (~30s). Subsequent calls use cache.

---

## Module 2 — `apply_registration_to_bboxes.py`

Applies the best affine transform found by `stall_affine_search.py` to all normalised bounding box columns in the scaled CSV. Produces registered coordinates in the canonical stall frame.

**Depends on:** `stall_affine_search.py` (must be in the same directory).

### Workflow

1. Set `FLOATING_STALL` normalised stall corners from `select_check_points.py` output.
2. Set `STANDARD_STALL` canonical stall shape (default: `x ∈ [-0.8, 0.8]`, `y ∈ [-1, 1]`).
3. Run `find_local_minima()` to get the best transform parameters.
4. Apply the 7-DOF affine to every bounding box corner set.
5. Save the registered CSV.

### Inputs / Outputs

| | File | Description |
|--|------|-------------|
| **Input** | `*_scaled.csv` | Output of `select_check_points.py` |
| **Output** | `*_scaled_reg.csv` | Same table with added `*_reg` columns per body part |

### Added columns (per prefix: `body box`, `head box`, `snout box`)

| Column | Description |
|--------|-------------|
| `Ln_reg`, `Tn_reg` | Registered top-left corner |
| `Wn_reg`, `Hn_reg` | Registered width and height |
| `AR_reg` | Aspect ratio after registration |
| `logAR_reg` | Log aspect ratio after registration |

### Requirements

```bash
pip install numpy pandas numba openpyxl
```

---

## Pipeline Position
<pre>
select_check_points.py
        │
        ▼
*_scaled.csv
        │
        ▼
stall_affine_search.py
(called by apply_registration_to_bboxes.py)
        │
        ▼
*_scaled_reg.csv
        │
        ▼
HMM fitting & evaluation
</pre>


---

## Notes

- `stall_affine_search.py` must be in the same directory as `apply_registration_to_bboxes.py`.
- Grid resolution `Nr=7` → `7^7 = 823,543` cells (~6.6 MB). Increase for finer search at higher memory/time cost.

---

*Part of the WELL-E animal welfare research pipeline — UQAM / McGill.*
