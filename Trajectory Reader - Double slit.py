import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import gridspec
import matplotlib as mpl
from matplotlib.patches import Rectangle, Circle
# ================================================================
# ======================= USER SETTINGS ===========================
# ================================================================

# Windows
FILEPATH = r"C:\Users\jacob\OneDrive\Documents\Physics - Bristol\Year 3\3rd Year Research Project\Best Trajectories\trajectories_double_d=17mm_l=14mm.csv"
OUTPUT_FOLDER = r"C:\Users\jacob\OneDrive\Documents\Physics - Bristol\Year 3\3rd Year Research Project\Final Report\Plots for report"

_filepath_name = os.path.basename(FILEPATH).lower()
if "d=13mm" in _filepath_name:
    D_MM = 13.0
    SLIT_WIDTH = 10.0
elif "d=17mm" in _filepath_name:
    D_MM = 17.0
    SLIT_WIDTH = 14.0
else:
    raise ValueError(f"Could not infer double-slit geometry from FILEPATH: {FILEPATH}")
LAMBDA_MM = 5.193

# Mac
#FILEPATH = "/Applications/OneDrive - University of Bristol/3rd Year/Research Project/Best Trajectories/trajectories_double_d=13mm.csv"
#OUTPUT_FOLDER = "/Applications/OneDrive - University of Bristol/3rd Year/Research Project/Saved Plots"

# Mac-mini
#FILEPATH = "/Users/harrytonge/Library/CloudStorage/OneDrive-UniversityofBristol(2)/3rd Year/Research Project/best trajectories/trajectories_double_d=13mm_2.csv"
#OUTPUT_FOLDER = "/Users/harrytonge/Library/CloudStorage/OneDrive-UniversityofBristol(2)/3rd Year/Research Project/Saved Plots"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===================== UNIT CALIBRATION =====================
MM_PER_PX = 0.11428571   # mm per pixel
FPS = 9.17               # camera FPS
# ============================================================

# ----------------- Slit geometry (PIXELS internally) -----------------
X_SLIT_IN = 250
Y_SLIT_CENTER_PX = 307

# ----------- Red/Blue classification (kept for trajectory plots) -----------
X_CLASSIFY = X_SLIT_IN - 40
COLOR_A = "blue"
COLOR_B = "red"
Y_SPLIT = Y_SLIT_CENTER_PX

# ================================================================
# ==================== CIRCULAR SEPARATOR CUT =====================
# ================================================================
USE_SEPARATOR_CUT = True

SEP_CX_PX = 230   # <-- EDIT
SEP_CY_PX = 310   # <-- EDIT

SEP_DIAMETER_MM = 3.0
SEP_RADIUS_PX = (SEP_DIAMETER_MM / 2.0) / MM_PER_PX

SEP_PAD_MM = 0.0
SEP_PAD_PX = SEP_PAD_MM / MM_PER_PX

# ================================================================
# ================= DOUBLE-SLIT DRAWING SETTINGS =================
# ================================================================

SHOW_DOUBLE_SLIT_BLOCKS = True

# Plate position in x (all in pixels)
DOUBLE_SLIT_PLATE_THICKNESS_PX = 44
DOUBLE_SLIT_X_OFFSET_PX = 45   # how far left of X_SLIT_IN the plate starts

# Slit geometry in y (pixels)
DOUBLE_SLIT_CENTER_PX = Y_SLIT_CENTER_PX

# Each slit opening height, hand-tuned in pixels to match the camera geometry
if D_MM == 13.0:
    DOUBLE_SLIT_OPENING_PX = 70
elif D_MM == 17.0:
    DOUBLE_SLIT_OPENING_PX = 98
else:
    raise ValueError(f"Could not infer double-slit opening in pixels for D_MM={D_MM}")

# Centre-to-centre separation of the two slit openings
DOUBLE_SLIT_SEPARATION_PX = int(round(D_MM / MM_PER_PX))

# Appearance
DOUBLE_SLIT_FACECOLOR = "red"
DOUBLE_SLIT_ALPHA = 0.22
DOUBLE_SLIT_EDGECOLOR = "firebrick"
DOUBLE_SLIT_EDGEWIDTH = 1.5

# ----------------- Track length / quality -----------------
MIN_TRACK_LENGTH = 25

MAX_BACKTRACK_FRAC = 0.20

MAX_ABS_THETA = 75
FLAT_DY_THRESH = 6.0
MAX_RMSE_FLAT = 2.5
MIN_R2 = 0.85

# Post-slit plotting window (PIXELS internally)
X_POST_MAX = X_SLIT_IN + 200

# Histogram bins
NBINS_THETA = 60
NBINS_YI = 40
NBINS_Y = 40

# Aesthetics
LINEWIDTH_PRE = 0.8
LINEWIDTH_POST = 1.2
ALPHA_PRE = 0.18
ALPHA_POST = 0.20
RAW_LW = 1.0
RAW_ALPHA = 1.0

# Where to sample y for the "position histogram" (slit entrance)
X_SLIT_ENTRANCE = X_SLIT_IN - 40   # <-- EDIT entrance offset
X_HIST = X_SLIT_ENTRANCE

# Extra y-position histogram plane: X_HIST_POST_MM after slit
X_HIST_POST_MM = 5
X_HIST_POST_PX = int(round(X_SLIT_IN + X_HIST_POST_MM / MM_PER_PX))

# ================================================================
# ===================== SPEED FILTER SETTINGS =====================
# ================================================================
# NEW: compute AVERAGE speed in the pre-slit section and keep only
# if SPEED_MIN <= v_avg <= SPEED_MAX (mm/s).

USE_T_S_IF_AVAILABLE = True

# Pre-slit window where speed is computed (pixels)
SPEED_X_MIN = X_SLIT_IN - 120   # <-- EDIT
SPEED_X_MAX = X_SLIT_IN - 10    # <-- EDIT

# Use mean speed (average) over the window
SPEED_STAT = "mean"

# Reject tracks with too many zero-steps
MAX_ZERO_STEP_FRAC = 0.50

# Fixed speed band in mm/s (EDIT THESE)
SPEED_MIN = 9.0   # mm/s
SPEED_MAX = 14.0   # mm/s

# ================================================================
# ============ DEFLECTION ANGLE: "FAR-FIELD" PLATEAU (3) ==========
# ================================================================
PLATEAU_X_START_OFFSET = 0
PLATEAU_X_END_OFFSET   = 50
WINDOW_WIDTH_PX = 50
WINDOW_STEP_PX  = 10
MIN_POINTS_PER_WINDOW = 4

PLATEAU_NEED_CONSEC_WINDOWS = 3
THETA_PLATEAU_DTH = 2.0
THETA_PLATEAU_MAX_STD = 2.5

FALLBACK_TO_FIXED_WINDOW = True
THETA_X_START_OFFSET = 0
THETA_X_END_OFFSET   = 50
THETA_X_MIN = X_SLIT_IN + THETA_X_START_OFFSET
THETA_X_MAX = X_SLIT_IN + THETA_X_END_OFFSET
MIN_THETA_POINTS_FALLBACK = 4

# ================================================================
# =================== ENTRY DISTRIBUTION (4b) =====================
# ================================================================
DO_YI_HIST = True
DO_REWEIGHTED_THETA_HIST = True
REWEIGHT_YI_BINS = 12

# ================================================================
# =================== SLIT / FARADAY PARAMETERS (4e) ==============
# ================================================================
SLIT_WIDTH_MM = SLIT_WIDTH
LAMBDA_F_MM = None

# ================================================================
# ============================ LOADING ============================
# ================================================================
def load_trajectories(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)
    elif ext == ".csv":
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    req = {"droplet_id", "frame", "x_px", "y_px"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "run_id" not in df.columns:
        df["run_id"] = "unknown"

    for c in ["frame", "x_px", "y_px", "droplet_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "t_s" in df.columns:
        df["t_s"] = pd.to_numeric(df["t_s"], errors="coerce")

    df = df.dropna(subset=["frame", "x_px", "y_px", "droplet_id"])
    df = df.sort_values(["run_id", "droplet_id", "frame"]).reset_index(drop=True)
    return df

# ================================================================
# ======================= GEOMETRY HELPERS ========================
# ================================================================
def y_at_x_first_crossing(x, y, x_target):
    x = np.asarray(x); y = np.asarray(y)
    for i in range(len(x) - 1):
        x0, x1 = x[i], x[i + 1]
        if (x0 - x_target) * (x1 - x_target) <= 0:
            if x1 == x0:
                return y[i]
            t = (x_target - x0) / (x1 - x0)
            return y[i] + t * (y[i + 1] - y[i])
    return None

def wrap_to_90(angle_deg):
    return ((angle_deg + 90) % 180) - 90

def line_fit_r2(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 2 or np.ptp(x) < 1e-12:
        return None, None, None

    m, c = np.polyfit(x, y, 1)
    yhat = m * x + c
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot < 1e-12:
        r2 = 1.0 if ss_res < 1e-12 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    return m, c, r2

def angle_in_x_window(x, y, x_min, x_max, min_points=6, max_backtrack_frac=None):
    x = np.asarray(x); y = np.asarray(y)
    mask = (x >= x_min) & (x <= x_max)
    npts = int(np.count_nonzero(mask))
    if npts < min_points:
        return None, None, None, None, npts, "not_enough_points"

    xw = x[mask]
    yw = y[mask]

    if max_backtrack_frac is not None and npts >= 3:
        dx = np.diff(xw)
        if np.mean(dx < 0) > max_backtrack_frac:
            return None, None, None, None, npts, "too_much_backtrack"

    m, c, r2 = line_fit_r2(xw, yw)
    if m is None or r2 is None:
        return None, None, None, None, npts, "fit_failed"

    yhat = m * xw + c
    ss_res = np.sum((yw - yhat) ** 2)
    rmse = np.sqrt(ss_res / max(1, len(yw)))
    dy = np.ptp(yw)

    theta = np.degrees(np.arctan(m))
    return wrap_to_90(theta), r2, rmse, dy, npts, "ok"

def classify_red_blue(x, y):
    y_ref = y_at_x_first_crossing(x, y, X_CLASSIFY)
    if y_ref is None:
        y_ref = y[0]
    return COLOR_A if y_ref < Y_SPLIT else COLOR_B

def track_hits_separator(x, y):
    if not USE_SEPARATOR_CUT:
        return False
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = float(SEP_RADIUS_PX + SEP_PAD_PX)
    dx = x - float(SEP_CX_PX)
    dy = y - float(SEP_CY_PX)
    return bool(np.any((dx*dx + dy*dy) <= (r*r)))

# ================================================================
# ========================= SPEED HELPERS =========================
# ================================================================
def compute_track_speed(d: pd.DataFrame,
                        x_min=None, x_max=None,
                        use_t_s_if_available=True,
                        fps=None,
                        stat="mean",
                        max_zero_step_frac=None,
                        mm_per_px=1.0):
    """
    Returns (speed_value_mm_s, "mm/s", status_string)
    speed_value is mean/median of step-speeds within [x_min, x_max].
    """
    g = d.sort_values("frame")
    x = g["x_px"].to_numpy()
    y = g["y_px"].to_numpy()

    mask = np.ones(len(g), dtype=bool)
    if x_min is not None:
        mask &= (x >= x_min)
    if x_max is not None:
        mask &= (x <= x_max)

    if np.count_nonzero(mask) < 3:
        return None, "mm/s", "no_points_in_speed_window"

    xw = x[mask]
    yw = y[mask]

    dt = None
    if use_t_s_if_available and ("t_s" in g.columns):
        tw = g.loc[mask, "t_s"].to_numpy()
        if np.isfinite(tw).all() and np.ptp(tw) > 0:
            dt = np.diff(tw).astype(float)

    if dt is None:
        fr = g.loc[mask, "frame"].to_numpy().astype(float)
        dt = np.diff(fr)
        if fps is None or fps <= 0:
            return None, "mm/s", "fps_missing"
        dt = dt / float(fps)

    dist_px = np.hypot(np.diff(xw), np.diff(yw))
    dist_mm = dist_px * float(mm_per_px)

    good = dt > 0
    if np.count_nonzero(good) < 1:
        return None, "mm/s", "bad_dt"

    sp_steps = dist_mm[good] / dt[good]
    if len(sp_steps) < 2:
        return None, "mm/s", "not_enough_steps"

    if max_zero_step_frac is not None:
        zero_frac = np.mean(dist_mm[good] <= 1e-12)
        if zero_frac > max_zero_step_frac:
            return None, "mm/s", "too_many_zero_steps"

    if stat == "median":
        return float(np.nanmedian(sp_steps)), "mm/s", "ok"
    return float(np.nanmean(sp_steps)), "mm/s", "ok"

# ================================================================
# =========== FAR-FIELD DEFLECTION ANGLE VIA PLATEAU (3) ===========
# ================================================================
def angle_from_plateau_postslit(x, y):
    x = np.asarray(x); y = np.asarray(y)

    x_start = X_SLIT_IN + PLATEAU_X_START_OFFSET
    x_end   = X_SLIT_IN + PLATEAU_X_END_OFFSET

    thetas, r2s, rmses, dys, x_centres, statuses = [], [], [], [], [], []

    w = WINDOW_WIDTH_PX
    step = WINDOW_STEP_PX

    for xs in np.arange(x_start, x_end - w + 1, step):
        xe = xs + w
        th, r2, rmse, dy, npts, st = angle_in_x_window(
            x, y, xs, xe,
            min_points=MIN_POINTS_PER_WINDOW,
            max_backtrack_frac=MAX_BACKTRACK_FRAC
        )
        x_centres.append(xs + 0.5 * w)
        thetas.append(th); r2s.append(r2); rmses.append(rmse); dys.append(dy); statuses.append(st)

    x_centres = np.array(x_centres)
    thetas = np.array(thetas, dtype=float)
    r2s = np.array(r2s, dtype=float)
    rmses = np.array(rmses, dtype=float)
    dys = np.array(dys, dtype=float)
    statuses = np.array(statuses)

    valid = np.isfinite(thetas) & (statuses == "ok")
    if np.count_nonzero(valid) < PLATEAU_NEED_CONSEC_WINDOWS:
        return None, {"x_centres": x_centres, "thetas": thetas, "valid": valid}, "not_enough_windows"

    good_line = np.zeros_like(valid, dtype=bool)
    for i in range(len(valid)):
        if not valid[i]:
            continue
        dy = dys[i]
        r2 = r2s[i]
        rmse = rmses[i]
        if np.isfinite(dy) and dy < FLAT_DY_THRESH:
            if np.isfinite(rmse) and rmse <= MAX_RMSE_FLAT:
                good_line[i] = True
        else:
            if np.isfinite(r2) and r2 >= MIN_R2:
                good_line[i] = True

    idx = np.where(good_line)[0]
    if len(idx) < PLATEAU_NEED_CONSEC_WINDOWS:
        return None, {"x_centres": x_centres, "thetas": thetas, "valid": good_line}, "no_plateau"

    for start_i in range(len(idx) - PLATEAU_NEED_CONSEC_WINDOWS + 1):
        run = idx[start_i:start_i + PLATEAU_NEED_CONSEC_WINDOWS]
        if np.any(np.diff(run) != 1):
            continue
        th_run = thetas[run]
        if not np.all(np.isfinite(th_run)):
            continue
        if np.max(np.abs(np.diff(th_run))) <= THETA_PLATEAU_DTH and np.std(th_run) <= THETA_PLATEAU_MAX_STD:
            theta_plateau = float(np.median(th_run))
            return theta_plateau, {
                "x_centres": x_centres,
                "thetas": thetas,
                "good_line": good_line,
                "plateau_indices": run
            }, "ok"

    return None, {"x_centres": x_centres, "thetas": thetas, "good_line": good_line}, "no_plateau"

# ================================================================
# ============================ FILTERS =============================
# ================================================================
def passes_basic_filters_reason(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < MIN_TRACK_LENGTH:
        return False, "short"
    yi_abs = y_at_x_first_crossing(x, y, X_SLIT_IN)
    if yi_abs is None:
        return False, "no_cross_xslit"
    return True, "ok"

# ================================================================
# ============================ PLOTS ==============================
# ================================================================
def style_axes(ax):
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

def weighted_hist_density(data, bins, weights):
    counts, edges = np.histogram(data, bins=bins, weights=weights, density=False)
    widths = np.diff(edges)
    area = np.sum(counts * widths)
    density = (counts / area) if area > 0 else counts * 0.0
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, density, edges

def plot_raw_trajectories_coloured_by_speed(df, speed_by_key, group_cols,
                                            mm_per_px, x_slit_in_px,
                                            y_slit_center_px, y_split_px=None,
                                            title="ALL trajectories (raw) coloured by speed (5 bins)",
                                            use_quintiles=True):
    """
    Plot ALL trajectories (raw) coloured by per-track mean pre-slit speed.
    Speeds are split into 5 bins (quintiles by default).
    Tracks with invalid speed are drawn in light grey.
    """
    # Collect valid speeds
    speeds = []
    for key in df.groupby(group_cols, sort=False).groups.keys():
        v = speed_by_key.get(key, None)
        if v is None or not np.isfinite(v):
            continue
        speeds.append(float(v))

    speeds = np.asarray(speeds, dtype=float)
    if speeds.size < 5:
        print("Not enough valid per-track speeds to build 5 bins.")
        return

    # 5 regions
    if use_quintiles:
        edges = np.quantile(speeds, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # if duplicates (many identical speeds), fallback to equal-width
        if np.unique(edges).size < 6:
            edges = np.linspace(np.min(speeds), np.max(speeds), 6)
    else:
        edges = np.linspace(np.min(speeds), np.max(speeds), 6)

    # Colormap & norm
    cmap = plt.get_cmap("viridis", len(edges) - 1)
    norm = mpl.colors.BoundaryNorm(edges, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot tracks
    for key, d in df.groupby(group_cols, sort=False):
        x_mm = d["x_px"].to_numpy(dtype=float) * mm_per_px
        y_mm = d["y_px"].to_numpy(dtype=float) * mm_per_px

        v = speed_by_key.get(key, None)
        if v is None or not np.isfinite(v):
            ax.plot(x_mm, y_mm, color="0.75", lw=RAW_LW, alpha=0.95)
        else:
            ax.plot(x_mm, y_mm, color=cmap(norm(v)), lw=RAW_LW, alpha=0.95)

    # Guides
    ax.axvline(x_slit_in_px * mm_per_px, color="k", ls="--", lw=1.2, alpha=0.8, label="slit x")
    ax.axhline(y_slit_center_px * mm_per_px, color="k", ls=":", lw=1.0, alpha=0.6, label="slit centre")

    if y_split_px is not None:
        ax.axhline(y_split_px * mm_per_px, color="gold", ls="--", lw=1.2, alpha=0.9, label="launch split")

    # Optional separator circle shown for context (not used for filtering here)
    if USE_SEPARATOR_CUT:
        circ = plt.Circle((SEP_CX_PX * mm_per_px, SEP_CY_PX * mm_per_px),
                          (SEP_RADIUS_PX + SEP_PAD_PX) * mm_per_px,
                          fill=False, lw=2.0, ls="--", alpha=0.9,
                          label="separator circle")
        ax.add_patch(circ)

    ax.invert_yaxis()
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    style_axes(ax)

    # Colorbar with bin edges
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Mean pre-slit speed (mm/s)")
    cbar.set_ticks(edges)
    cbar.ax.set_yticklabels([f"{e:.2f}" for e in edges])

    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "Double slit all trajectories raw coloured by mean pre-slit speed.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    
def add_double_slit_blocks(ax, mm_per_px, x_slit_in_px):
    """
    Draw shaded double-slit barrier blocks on an existing axis.
    Top and bottom are rectangular barriers.
    Central separator is circular.
    """

    if not SHOW_DOUBLE_SLIT_BLOCKS:
        return

    x_slit_mm = x_slit_in_px * mm_per_px
    plate_thickness_mm = DOUBLE_SLIT_PLATE_THICKNESS_PX * mm_per_px
    x_offset_mm = DOUBLE_SLIT_X_OFFSET_PX * mm_per_px

    y_centre_mm = DOUBLE_SLIT_CENTER_PX * mm_per_px
    opening_mm = DOUBLE_SLIT_OPENING_PX * mm_per_px
    separation_mm = DOUBLE_SLIT_SEPARATION_PX * mm_per_px

    # Centres of upper and lower slit openings
    y_upper_centre_mm = y_centre_mm - separation_mm / 2.0
    y_lower_centre_mm = y_centre_mm + separation_mm / 2.0

    # Edges of the two open slits
    upper_top_mm = y_upper_centre_mm - opening_mm / 2.0
    upper_bottom_mm = y_upper_centre_mm + opening_mm / 2.0

    lower_top_mm = y_lower_centre_mm - opening_mm / 2.0
    lower_bottom_mm = y_lower_centre_mm + opening_mm / 2.0

    # Axis limits
    y0, y1 = ax.get_ylim()
    y_top = min(y0, y1)
    y_bottom = max(y0, y1)

    # Plate x-position
    x_left_mm = x_slit_mm - x_offset_mm
    x_right_mm = x_left_mm + plate_thickness_mm

    # ------------------------------------------------------------
    # Top rectangular barrier
    # ------------------------------------------------------------
    if upper_top_mm > y_top:
        ax.add_patch(Rectangle(
            (x_left_mm, y_top),
            plate_thickness_mm,
            upper_top_mm - y_top,
            facecolor=DOUBLE_SLIT_FACECOLOR,
            edgecolor="none",
            alpha=DOUBLE_SLIT_ALPHA,
            zorder=0.1
        ))

        ax.plot([x_left_mm, x_left_mm], [y_top, upper_top_mm],
                color=DOUBLE_SLIT_EDGECOLOR, lw=DOUBLE_SLIT_EDGEWIDTH, zorder=2)
        ax.plot([x_right_mm, x_right_mm], [y_top, upper_top_mm],
                color=DOUBLE_SLIT_EDGECOLOR, lw=DOUBLE_SLIT_EDGEWIDTH, zorder=2)
        ax.plot([x_left_mm, x_right_mm], [upper_top_mm, upper_top_mm],
                color=DOUBLE_SLIT_EDGECOLOR, lw=DOUBLE_SLIT_EDGEWIDTH, zorder=2)

    # ------------------------------------------------------------
    # Bottom rectangular barrier
    # ------------------------------------------------------------
    if y_bottom > lower_bottom_mm:
        ax.add_patch(Rectangle(
            (x_left_mm, lower_bottom_mm),
            plate_thickness_mm,
            y_bottom - lower_bottom_mm,
            facecolor=DOUBLE_SLIT_FACECOLOR,
            edgecolor="none",
            alpha=DOUBLE_SLIT_ALPHA,
            zorder=0.1
        ))

        ax.plot([x_left_mm, x_left_mm], [lower_bottom_mm, y_bottom],
                color=DOUBLE_SLIT_EDGECOLOR, lw=DOUBLE_SLIT_EDGEWIDTH, zorder=2)
        ax.plot([x_right_mm, x_right_mm], [lower_bottom_mm, y_bottom],
                color=DOUBLE_SLIT_EDGECOLOR, lw=DOUBLE_SLIT_EDGEWIDTH, zorder=2)
        ax.plot([x_left_mm, x_right_mm], [lower_bottom_mm, lower_bottom_mm],
                color=DOUBLE_SLIT_EDGECOLOR, lw=DOUBLE_SLIT_EDGEWIDTH, zorder=2)

    # Circular central separator
    sep_cx_mm = SEP_CX_PX * mm_per_px
    sep_cy_mm = SEP_CY_PX * mm_per_px
    sep_r_mm = SEP_RADIUS_PX * mm_per_px
    
    # filled circle
    circ_fill = Circle(
        (sep_cx_mm, sep_cy_mm),
        sep_r_mm,
        facecolor=DOUBLE_SLIT_FACECOLOR,
        edgecolor="none",
        alpha=DOUBLE_SLIT_ALPHA,
        zorder=1.5
    )
    ax.add_patch(circ_fill)
    
    # outline circle
    circ_edge = Circle(
        (sep_cx_mm, sep_cy_mm),
        sep_r_mm,
        facecolor="none",
        edgecolor="firebrick",
        linewidth=2.5,
        alpha=1.0,
        zorder=3
    )
    ax.add_patch(circ_edge)
        
# ================================================================
# ============================== MAIN =============================
# ================================================================
def main():
    df = load_trajectories(FILEPATH)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n================ SLIT / FARADAY (4e) ================")
    print(f"Slit width L = {SLIT_WIDTH_MM} mm")
    if LAMBDA_F_MM is None:
        print("LAMBDA_F_MM is not set.")
    else:
        print(f"Faraday wavelength lambdaF = {LAMBDA_F_MM} mm -> L/lambdaF = {SLIT_WIDTH_MM / LAMBDA_F_MM:.3f}")
    print(f"Calibration: {MM_PER_PX} mm/px")
    print(f"FPS: {FPS} Hz")
    print("======================================================\n")

    group_cols = ["run_id", "droplet_id"]
    n_tracks_total = df.groupby(group_cols).ngroups

    # ---------------- Plot 0: ALL trajectories raw ----------------
    fig, ax = plt.subplots(figsize=(7, 6))
    for (run_id, droplet_id), d in df.groupby(group_cols, sort=False):
        x = d["x_px"].to_numpy()
        y = d["y_px"].to_numpy()
        col = classify_red_blue(x, y)
        ax.plot(x * MM_PER_PX, y * MM_PER_PX, color=col, lw=RAW_LW, alpha=RAW_ALPHA)

    ax.axvline(X_SLIT_IN * MM_PER_PX, color="k", ls="--", lw=1.2, alpha=0.8, label="slit x")
    ax.axhline(Y_SLIT_CENTER_PX * MM_PER_PX, color="k", ls=":", lw=1.0, alpha=0.6, label="slit centre")
    ax.axhline(Y_SPLIT * MM_PER_PX, color="gold", ls="--", lw=1.2, alpha=0.9, label="launch split (classifier)")
    ax.invert_yaxis()
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"ALL trajectories (raw, N={n_tracks_total})")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "Double slit all trajectories raw.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================================
    # PASS 1: compute average pre-slit speed per track (mm/s)
    # ============================================================
    speed_by_key = {}
    speed_reject = {}

    for key, d in df.groupby(group_cols, sort=False):
        sp, units, status = compute_track_speed(
            d,
            x_min=SPEED_X_MIN,
            x_max=SPEED_X_MAX,
            use_t_s_if_available=USE_T_S_IF_AVAILABLE,
            fps=FPS,
            stat=SPEED_STAT,  # mean
            max_zero_step_frac=MAX_ZERO_STEP_FRAC,
            mm_per_px=MM_PER_PX
        )
        if status != "ok":
            speed_reject[status] = speed_reject.get(status, 0) + 1
            speed_by_key[key] = None
        else:
            speed_by_key[key] = sp

    all_speeds = np.array([v for v in speed_by_key.values() if v is not None and np.isfinite(v)], dtype=float)
    if len(all_speeds) == 0:
        print("No valid speeds computed — check t_s / FPS / speed window.")
        return

    vmin = float(SPEED_MIN)
    vmax = float(SPEED_MAX)

    print("\n================ SPEED SUMMARY ================")
    print("Speed units: mm/s")
    print(f"Speed computed in pre-slit x-window (px): [{SPEED_X_MIN}, {SPEED_X_MAX}]")
    print(f"Speed statistic: {SPEED_STAT} (average step-speed)")
    print(f"Computed speeds for {len(all_speeds)} tracks.")
    print(f"Speed percentiles (10/50/90): "
          f"{np.percentile(all_speeds,10):.3f}, {np.percentile(all_speeds,50):.3f}, {np.percentile(all_speeds,90):.3f} mm/s")
    print(f"Speed filter: keep [{vmin:.3f}, {vmax:.3f}] mm/s")
    if speed_reject:
        print("Speed computation rejects:")
        for k in sorted(speed_reject.keys()):
            print(f"  {k:>28s} : {speed_reject[k]}")
    print("==============================================\n")
    
    # ============================================================
    # NEW Plot: ALL (unfiltered) trajectories coloured by speed (5 bins)
    # ============================================================
    plot_raw_trajectories_coloured_by_speed(
        df=df,
        speed_by_key=speed_by_key,
        group_cols=group_cols,
        mm_per_px=MM_PER_PX,
        x_slit_in_px=X_SLIT_IN,
        y_slit_center_px=Y_SLIT_CENTER_PX,
        y_split_px=Y_SPLIT,
        title="ALL trajectories (raw) coloured by mean pre-slit speed (5 bins)",
        use_quintiles=True  # set False for equal-width bins
    )
    # Speed histogram
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(all_speeds, bins=40, alpha=0.85)
    ax.axvline(vmin, color="k", ls="--", lw=1.2, alpha=0.8, label="keep min")
    ax.axvline(vmax, color="k", ls="--", lw=1.2, alpha=0.8, label="keep max")
    ax.set_xlabel("Average pre-slit speed (mm/s)")
    ax.set_ylabel("Count")
    ax.set_title("Trajectory speed distribution (per-track, pre-slit mean)")
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "Double slit trajectory speed distribution.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================================
    # PASS 2: Filter + compute far-field theta via plateau
    # ============================================================
    reject = {}
    kept = []

    yi_list, theta_list, speed_list = [], [], []
    plateau_fail_count = 0

    for key, d in df.groupby(group_cols, sort=False):
        x = d["x_px"].to_numpy()
        y = d["y_px"].to_numpy()

        ok, reason = passes_basic_filters_reason(x, y)
        if not ok:
            reject[reason] = reject.get(reason, 0) + 1
            continue

        if track_hits_separator(x, y):
            reject["hits_separator"] = reject.get("hits_separator", 0) + 1
            continue

        sp = speed_by_key.get(key, None)
        if sp is None or not np.isfinite(sp):
            reject["speed_invalid"] = reject.get("speed_invalid", 0) + 1
            continue
        if not (vmin <= sp <= vmax):
            reject["speed_out_of_band"] = reject.get("speed_out_of_band", 0) + 1
            continue

        yi_abs = y_at_x_first_crossing(x, y, X_SLIT_IN)
        if yi_abs is None:
            reject["no_cross_xslit"] = reject.get("no_cross_xslit", 0) + 1
            continue
        yi = yi_abs - Y_SLIT_CENTER_PX  # px

        theta, details, st = angle_from_plateau_postslit(x, y)

        if st != "ok":
            plateau_fail_count += 1
            reject[st] = reject.get(st, 0) + 1

            if FALLBACK_TO_FIXED_WINDOW:
                th2, r2, rmse, dy2, npts, st2 = angle_in_x_window(
                    x, y, THETA_X_MIN, THETA_X_MAX,
                    min_points=MIN_THETA_POINTS_FALLBACK,
                    max_backtrack_frac=MAX_BACKTRACK_FRAC
                )
                if st2 != "ok":
                    reject["fallback_failed"] = reject.get("fallback_failed", 0) + 1
                    continue

                if MAX_ABS_THETA is not None and abs(th2) > MAX_ABS_THETA:
                    reject["theta_outlier"] = reject.get("theta_outlier", 0) + 1
                    continue

                if dy2 is not None and dy2 < FLAT_DY_THRESH:
                    if rmse is None or rmse > MAX_RMSE_FLAT:
                        reject["flat_but_noisy"] = reject.get("flat_but_noisy", 0) + 1
                        continue
                else:
                    if r2 is None or r2 < MIN_R2:
                        reject["r2_too_low"] = reject.get("r2_too_low", 0) + 1
                        continue

                theta = th2
                r2_like = r2
                theta_status = "fallback_fixed_window"
            else:
                continue
        else:
            r2_like = np.nan
            theta_status = "plateau"

        if MAX_ABS_THETA is not None and abs(theta) > MAX_ABS_THETA:
            reject["theta_outlier"] = reject.get("theta_outlier", 0) + 1
            continue

        col = classify_red_blue(x, y)
        kept.append((x, y, yi, theta, col, r2_like, sp, theta_status))
        yi_list.append(yi); theta_list.append(theta); speed_list.append(sp)
        reject["ok"] = reject.get("ok", 0) + 1

    print("\nReject counts (why tracks were excluded):")
    for k in sorted(reject.keys()):
        print(f"  {k:>34s} : {reject[k]}")

    N = len(kept)
    print(f"\nKept N={N} tracks (speed in [{vmin:.3f}, {vmax:.3f}] mm/s).")
    if N == 0:
        print("No tracks kept. Try widening speed band or changing speed window.")
        return

    yi_arr_px = np.array(yi_list, dtype=float)
    yi_arr_mm = yi_arr_px * MM_PER_PX
    th_arr = np.array(theta_list, dtype=float)
    sp_arr = np.array(speed_list, dtype=float)

    print(f"Plateau failures (then fallback if enabled): {plateau_fail_count}")
    print(f"Kept speed: mean={np.mean(sp_arr):.3f}, median={np.median(sp_arr):.3f} mm/s\n")
    # ============================================================
    # Find the physical centre theta_c (peak), then reflect about it
    # ============================================================
    
    theta_data = th_arr[np.isfinite(th_arr)]
    
    # Use the same binning as your theta histogram for consistency
    theta_bins = np.linspace(-90, 90, NBINS_THETA + 1)
    centres = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    dtheta = centres[1] - centres[0]
    
    # Histogram density
    P, _ = np.histogram(theta_data, bins=theta_bins, density=True)
    
    def gauss_smooth_1d(y, sigma_deg, dtheta_deg):
        """Gaussian smoothing in 'degrees' using convolution on the binned histogram."""
        if sigma_deg is None or sigma_deg <= 0:
            return y
        sigma_bins = sigma_deg / dtheta_deg
        r = int(np.ceil(4 * sigma_bins))
        k = np.arange(-r, r + 1)
        g = np.exp(-0.5 * (k / sigma_bins) ** 2)
        g /= g.sum()
        return np.convolve(y, g, mode="same")
    
    # Smooth enough to kill bin-noise but not smear lobes
    P_smooth = gauss_smooth_1d(P, sigma_deg=3.0, dtheta_deg=dtheta)
    
    # ---- main-peak centre estimate ----
    # Option A (recommended): peak of smoothed histogram
    theta_c = float(centres[np.argmax(P_smooth)])
    
    # Option B (alternative): restrict search to a central window, e.g. ±25°
    # m_central = (centres >= -25) & (centres <= 25)
    # theta_c = float(centres[m_central][np.argmax(P_smooth[m_central])])
    
    print(f"Estimated histogram centre (theta_c) = {theta_c:+.2f} deg")
    
    # Reflect about theta_c: theta_ref = 2*theta_c - theta
    theta_ref = 2.0 * theta_c - theta_data
    
    # Pool the data with its reflection (NOT dividing by 2; this is a pooled dataset)
    theta_pooled = np.concatenate([theta_data, theta_ref])
    
    # If you want the "symmetrised PDF" explicitly, do it via histograms:
    P_pool, edges = np.histogram(theta_pooled, bins=theta_bins, density=True)
    # (density=True already normalises area to 1, but you can renormalise for safety)
    bin_w = np.diff(edges)
    area = np.sum(P_pool * bin_w)
    if area > 0:
        P_pool = P_pool / area
    # ============================================================
    # Histogram of y positions at x=X_HIST (kept) — RELATIVE to slit centre (mm)
    # ============================================================
    y_hist_rel_px = []
    for x, y, yi, theta, col, r2_like, sp, theta_status in kept:
        y_at_hist = y_at_x_first_crossing(x, y, X_HIST)
        if y_at_hist is not None:
            y_hist_rel_px.append(y_at_hist - Y_SLIT_CENTER_PX)

    y_hist_rel_px = np.array(y_hist_rel_px, dtype=float)
    y_hist_rel_mm = y_hist_rel_px * MM_PER_PX

    fig, ax = plt.subplots(figsize=(6, 5))
    if len(y_hist_rel_mm) == 0:
        ax.text(0.5, 0.5, f"No KEPT tracks cross x={X_HIST * MM_PER_PX:.2f} mm", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(y_hist_rel_mm, bins=NBINS_Y, alpha=0.85)
        ax.axvline(0, color="k", ls="--", lw=1.2, alpha=0.8, label="slit centre (0)")
        ax.set_xlabel(f"$y - y_{{\\rm slit}}$ at x = {X_HIST * MM_PER_PX:.2f} mm (mm)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Histogram of (y - slit centre) at slit entrance\n"
            f"(x={X_HIST * MM_PER_PX:.2f} mm, KEPT tracks, N={len(y_hist_rel_mm)})"
        )
        style_axes(ax)
        ax.legend(frameon=False)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "Double slit histogram of y positions at slit entrance.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================================
    # yi histogram at slit exit (mm)
    # ============================================================
    if DO_YI_HIST:
        fig, ax = plt.subplots(figsize=(6, 4.8))
        ax.hist(yi_arr_mm, bins=NBINS_YI, alpha=0.85)
        ax.axvline(0, color="k", ls="--", lw=1.2, alpha=0.8, label="slit centre (yi=0)")
        ax.set_xlabel("Impact parameter $y_i$ (mm; y_at_slit - slit_center)")
        ax.set_ylabel("Count")
        ax.set_title(f"Impact parameter distribution at slit exit (kept N={N})")
        style_axes(ax)
        ax.legend(frameon=False)
        fig.tight_layout()
        out_path = os.path.join(OUTPUT_FOLDER, "Double slit impact parameter distribution at slit exit.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.show()

    # ============================================================
    # Histogram of y positions at x = slit + X_HIST_POST_MM (kept), relative to slit centre
    # ============================================================
    y_post_rel_px = []
    for x, y, yi, theta, col, r2_like, sp, theta_status in kept:
        y_at_post = y_at_x_first_crossing(x, y, X_HIST_POST_PX)
        if y_at_post is not None:
            y_post_rel_px.append(y_at_post - Y_SLIT_CENTER_PX)

    y_post_rel_px = np.array(y_post_rel_px, dtype=float)
    y_post_rel_mm = y_post_rel_px * MM_PER_PX

    fig, ax = plt.subplots(figsize=(6, 5))
    if len(y_post_rel_mm) == 0:
        ax.text(0.5, 0.5, f"No KEPT tracks cross x={X_HIST_POST_PX * MM_PER_PX:.2f} mm",
                ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(y_post_rel_mm, bins=NBINS_Y, alpha=0.85)
        ax.axvline(0, color="k", ls="--", lw=1.2, alpha=0.8, label="slit centre (0)")
        ax.set_xlabel(f"$y - y_{{\\rm slit}}$ at x = {X_HIST_POST_PX * MM_PER_PX:.2f} mm (mm)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Histogram of (y - slit centre) at x = slit + {X_HIST_POST_MM:.0f} mm\n"
            f"(x={X_HIST_POST_PX * MM_PER_PX:.2f} mm, KEPT tracks, N={len(y_post_rel_mm)})"
        )
        style_axes(ax)
        ax.legend(frameon=False)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, f"Double slit histogram of y positions {X_HIST_POST_MM:.0f} mm after slit.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

        # ============================================================
        # Plot 1: Trajectories with hierarchy (kept only)
        # ============================================================
    fig, ax = plt.subplots(figsize=(7, 6))

    for x, y, yi, theta, col, r2_like, sp, theta_status in kept:
        pre = x <= X_SLIT_IN
        if np.any(pre):
            ax.plot(x[pre] * MM_PER_PX, y[pre] * MM_PER_PX,
                    color="0.3", lw=LINEWIDTH_PRE, alpha=ALPHA_PRE)

        post = (x >= X_SLIT_IN) & (x <= X_POST_MAX)
        if np.any(post):
            ax.plot(x[post] * MM_PER_PX, y[post] * MM_PER_PX,
                    color=col, lw=LINEWIDTH_POST, alpha=ALPHA_POST)

    ax.axvline(X_SLIT_IN * MM_PER_PX, color="k", ls="--", lw=1.2, alpha=0.8, label="Slit exit x")
    ax.axhline(Y_SPLIT * MM_PER_PX, color="gold", ls="--", lw=1.2, alpha=0.9, label="Launcher y")

    # if USE_SEPARATOR_CUT:
    #     circ = plt.Circle((SEP_CX_PX * MM_PER_PX, SEP_CY_PX * MM_PER_PX),
    #                       (SEP_RADIUS_PX + SEP_PAD_PX) * MM_PER_PX,
    #                       fill=False, lw=2.0, ls="--", alpha=0.9,
    #                       label="separator cut (circle)")
    #     ax.add_patch(circ)

    ax.invert_yaxis()
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_xlim(0,50)
    ax.set_ylim(55,15)
    ax.set_title(f"Droplet Trajectories (kept N={N})")
    style_axes(ax)

    # Add double-slit barrier blocks
    add_double_slit_blocks(ax, MM_PER_PX, X_SLIT_IN)

    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "Double slit kept trajectories.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    # ============================================================
    # Plot 2: Post-slit fan (slit-aligned)
    # ============================================================
    fig, ax = plt.subplots(figsize=(7, 6))
    for x, y, yi, theta, col, r2_like, sp, theta_status in kept:
        mask = (x >= X_SLIT_IN) & (x <= X_POST_MAX)
        xp = (x[mask] - X_SLIT_IN) * MM_PER_PX
        yp = (y[mask] - Y_SLIT_CENTER_PX) * MM_PER_PX
        if len(xp) >= 2:
            ax.plot(xp, yp, color=col, lw=1.2, alpha=0.75)

    ax.axvline(0, color="k", ls="--", lw=1.2, alpha=0.8)
    ax.axhline(0, color="k", ls=":", lw=1.0, alpha=0.7)
    ax.axvspan(PLATEAU_X_START_OFFSET * MM_PER_PX,
               PLATEAU_X_END_OFFSET * MM_PER_PX,
               alpha=0.10, label="Plateau search region")
    ax.invert_yaxis()
    ax.set_xlabel("x (mm) (shifted so slit at 0)")
    ax.set_ylabel("y (mm) (shifted so slit centre at 0)")
    ax.set_title("Post-slit Trajectories (slit-aligned)")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "Double slit post-slit fan slit-aligned.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================================
    # Plot 3: theta vs yi
    # ============================================================
    cols = np.array([k[4] for k in kept])
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in [COLOR_A, COLOR_B]:
        m = cols == c
        ax.scatter(th_arr[m], yi_arr_mm[m], s=35, alpha=0.9, label=c)

    ax.axvline(0, color="k", lw=1.0, alpha=0.6)
    ax.axhline(0, color="k", lw=1.0, alpha=0.6)
    ax.set_ylabel("Impact parameter $y_i$ (mm, relative to slit centre)")
    ax.set_xlabel("Deflection angle $\\theta$ (deg, from horizontal)")
    ax.set_title(f"Deflection Angle vs Impact Parameter (N={N})")
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "Double slit impact parameter vs deflection angle.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================================
    # Plot 4: theta histogram
    # ============================================================
    theta_bins = np.linspace(-90, 90, NBINS_THETA + 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(th_arr, bins=theta_bins, density=True, histtype="stepfilled", alpha=0.55)
    ax.set_xlim(-90, 90)
    ax.set_xlabel(r"$\theta$ (deg, from horizontal)")
    ax.set_ylabel(r"$P(\theta)$")
    ax.set_title(f"Deflection angle distribution (kept N={N})")
    style_axes(ax)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "Double slit deflection angle distribution.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================================
    # Plot 4: theta histogram + theory overlay (like ss2)
    # ============================================================

    theta_data = th_arr[np.isfinite(th_arr)]

    theta_bins = np.linspace(-90, 90, NBINS_THETA + 1)
    centres = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    dtheta = centres[1] - centres[0]

    def gauss_smooth_1d(y, sigma_deg, dtheta_deg):
        if sigma_deg is None or sigma_deg <= 0:
            return y
        sigma_bins = sigma_deg / dtheta_deg
        r = int(np.ceil(4 * sigma_bins))
        k = np.arange(-r, r + 1)
        g = np.exp(-0.5 * (k / sigma_bins) ** 2)
        g /= g.sum()
        return np.convolve(y, g, mode="same")

    # --- Estimate peak-centre theta_c from smoothed histogram (to align theory peak) ---
    P, _ = np.histogram(theta_data, bins=theta_bins, density=True)
    P_smooth = gauss_smooth_1d(P, sigma_deg=3.0, dtheta_deg=dtheta)

    central_mask = (centres >= -30) & (centres <= 30)  # avoids side-lobe peaks
    theta_c = float(centres[central_mask][np.argmax(P_smooth[central_mask])])
    print(f"[Plot 4] Peak-centre for theory alignment: theta_c = {theta_c:+.2f} deg")

    # ============================================================
    # Theory (same as your later section)
    # ============================================================
    L_MM = SLIT_WIDTH

    def sinc(x):
        y = np.ones_like(x, dtype=float)
        m = np.abs(x) > 1e-12
        y[m] = np.sin(x[m]) / x[m]
        return y

    def double_slit_amplitude(theta_deg, L_mm, d_mm, lam_mm, theta0_deg=0.0):
        th = np.deg2rad(theta_deg - theta0_deg)
        argL = np.pi * L_mm * np.sin(th) / lam_mm
        argd = np.pi * d_mm * np.sin(th) / lam_mm
        A = sinc(argL) * np.cos(argd)
        return np.abs(A)

    # Dense theta grid for smooth theory curve (in UNCENTERED coords)
    theta_dense = np.linspace(-90, 90, 6000)

    # Align the theoretical central maximum to theta_c
    A_dense = double_slit_amplitude(theta_dense, L_MM, D_MM, LAMBDA_MM, theta0_deg=theta_c)

    # Scale theory to match histogram peak height (same trick as ss2)
    A_dense = A_dense / (A_dense.max() if A_dense.max() > 0 else 1.0)
    P_peak = P.max() if P.size else 1.0
    A_dense_plot = A_dense * P_peak

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(theta_data, bins=theta_bins, density=True, histtype="stepfilled", alpha=0.55, label="Data")

    ax.plot(theta_dense, A_dense_plot, lw=2.4, label="Double-slit theory (scaled)", color ='black')

    ax.axvline(theta_c, color="k", ls="--", lw=1.1, alpha=0.7, label=r"peak centre $\theta_c$")
    ax.set_xlim(-90, 90)
    ax.set_xlabel(r"$\theta$ (deg, from horizontal)")
    ax.set_ylabel(r"$P(\theta)$ / scaled amplitude")
    ax.set_title(f"Deflection angle distribution (kept N={N}) with theory overlay")
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "Double slit deflection angle distribution with theory overlay.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    # ============================================================
    # Double-slit: reflect about measured peak, pool (one colour),
    # then recentre to 0 and overlay theory
    # ============================================================
    
    theta_data = th_arr[np.isfinite(th_arr)]
    
    # --- bins (use same as your plot) ---
    NBINS = 60
    theta_bins = np.linspace(-90, 90, NBINS + 1)
    centres = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    dtheta = centres[1] - centres[0]
    
    def gauss_smooth_1d(y, sigma_deg, dtheta_deg):
        if sigma_deg is None or sigma_deg <= 0:
            return y
        sigma_bins = sigma_deg / dtheta_deg
        r = int(np.ceil(4 * sigma_bins))
        k = np.arange(-r, r + 1)
        g = np.exp(-0.5 * (k / sigma_bins) ** 2)
        g /= g.sum()
        return np.convolve(y, g, mode="same")
    
    # --- estimate peak centre theta_c from smoothed histogram ---
    P, _ = np.histogram(theta_data, bins=theta_bins, density=True)
    P_smooth = gauss_smooth_1d(P, sigma_deg=3.0, dtheta_deg=dtheta)
    
    # Optional: restrict peak search to central region to avoid side-lobe dominance
    central_mask = (centres >= -30) & (centres <= 30)
    theta_c = float(centres[central_mask][np.argmax(P_smooth[central_mask])])
    
    print(f"Peak-centre for reflection: theta_c = {theta_c:+.2f} deg")
    
    # --- reflect about theta_c and pool ---
    theta_ref = 2.0 * theta_c - theta_data
    theta_pooled = np.concatenate([theta_data, theta_ref])
    
    # --- recentre pooled data so symmetry axis is at 0 deg ---
    theta_centered = theta_pooled - theta_c
    
    # --- histogram of centered pooled data (density) ---
    hist, edges = np.histogram(theta_centered, bins=theta_bins, density=True)
    hist_peak = hist.max() if hist.size else 1.0
    
    # ============================================================
    # Theory (Couder–Fort style amplitude)
    # ============================================================
    
    L_MM = SLIT_WIDTH
    
    def sinc(x):
        y = np.ones_like(x, dtype=float)
        m = np.abs(x) > 1e-12
        y[m] = np.sin(x[m]) / x[m]
        return y
    
    def double_slit_amplitude(theta_deg, L_mm, d_mm, lam_mm, theta0_deg=0.0):
        th = np.deg2rad(theta_deg - theta0_deg)
        argL = np.pi * L_mm * np.sin(th) / lam_mm
        argd = np.pi * d_mm * np.sin(th) / lam_mm
        A = sinc(argL) * np.cos(argd)
        return np.abs(A)
    
    # Smooth theory line on a dense grid in the *centered* coordinate
    theta_dense = np.linspace(-90, 90, 6000)
    A_dense = double_slit_amplitude(theta_dense, L_MM, D_MM, LAMBDA_MM, theta0_deg=0.0)
    A_dense = A_dense / (A_dense.max() if A_dense.max() > 0 else 1.0)
    A_dense_plot = A_dense * hist_peak
    
    # ============================================================
    # Plot: pooled (one colour) + theory overlay
    # ============================================================
    
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    
    # ONE colour histogram: this already includes original + reflected-about-peak
    ax.hist(theta_centered, bins=theta_bins, density=True, alpha=0.55,
            label=f"Data about peak (N={theta_centered.size})")
    
    ax.plot(theta_dense, A_dense_plot, lw=2.4,
            label=(r"Double-slit Theory"))
    
    ax.axvline(0, color="k", ls="--", lw=1.1, alpha=0.7)
    ax.set_xlim(-90, 90)
    ax.set_xlabel(r"$\theta$ (deg) (centred)")
    ax.set_ylabel(r"$P(\theta)$ / scaled amplitude")
    ax.set_title("Double-slit: pooled-about-peak (one colour), centred at 0, with theory overlay")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "Double slit pooled deflection angle distribution with theory overlay.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    # ============================================================
    # Diagnostics
    # ============================================================
    theta_statuses = [k[7] for k in kept]
    n_plateau = sum(s == "plateau" for s in theta_statuses)
    n_fallback = sum(s == "fallback_fixed_window" for s in theta_statuses)

    print("\n================ 4c DIAGNOSTIC (code-based) ================")
    print(f"  plateau theta measured: {n_plateau}")
    print(f"  fallback (near-slit window): {n_fallback}")
    print("============================================================\n")

    print("Done.")

if __name__ == "__main__":
    main()
