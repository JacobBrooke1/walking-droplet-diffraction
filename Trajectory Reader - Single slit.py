import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import gridspec
import matplotlib as mpl
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
# ================================================================
# ======================= USER SETTINGS ===========================
# ================================================================

# Windows
FILEPATH = r"C:\Users\jacob\OneDrive\Documents\Physics - Bristol\Year 3\3rd Year Research Project\Best Trajectories\trajectories_single_s=12mm - Too much.csv"
OUTPUT_FOLDER = r"C:\Users\jacob\OneDrive\Documents\Physics - Bristol\Year 3\3rd Year Research Project\Final Report\Plots for report"
SLIT_WIDTH = 12.0  # mm; change this to match the file in FILEPATH

# Mac
#FILEPATH = "/Applications/OneDrive - University of Bristol/3rd Year/Research Project/best trajectories/trajectories_single_s=12mm - Goldilocks.csv"
#OUTPUT_FOLDER = "/Applications/OneDrive - University of Bristol/3rd Year/Research Project/Saved Plots"

# Mac-mini
#FILEPATH = "/Users/harrytonge/Library/CloudStorage/OneDrive-UniversityofBristol(2)/3rd Year/Research Project/best trajectories/trajectories_single_s=12mm - Too little 2.csv"
#OUTPUT_FOLDER = "/Users/harrytonge/Library/CloudStorage/OneDrive-UniversityofBristol(2)/3rd Year/Research Project/Saved Plots"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ===================== UNIT CALIBRATION =====================
MM_PER_PX = 0.11428571   # mm per pixel
FPS = 9.17               # camera FPS
# ============================================================

# ----------------- Slit geometry (PIXELS internally) -----------------
X_SLIT_IN = 250
Y_SLIT_CENTER_PX = 307



# ----------- Red/Blue classification -----------
X_CLASSIFY = X_SLIT_IN - 40
COLOR_A = "blue"
COLOR_B = "red"
Y_SPLIT = Y_SLIT_CENTER_PX

# ================================================================
# ===================== SLIT DRAWING SETTINGS ====================
# ================================================================

SHOW_SLIT_BLOCKS = True

# Use the actual slit centre here if different from Y_SPLIT
Y_SLIT_CENTER_PX = Y_SPLIT

# Full opening height of the slit, derived from SLIT_WIDTH
SLIT_WIDTH_PX = int(round(SLIT_WIDTH / MM_PER_PX))

# Horizontal thickness of the shaded barrier
PLATE_THICKNESS_PX = 44

# How far left of x_slit_in the block starts
SLIT_X_OFFSET_PX = 45

# Appearance
SLIT_FACECOLOR = "red"
SLIT_ALPHA = 0.22
SLIT_EDGECOLOR = "firebrick"
SLIT_EDGEWIDTH = 1.5

# ----------------- Track length / quality -----------------
MIN_TRACK_LENGTH = 25

MAX_BACKTRACK_FRAC = 0.20

MAX_ABS_THETA = 75
FLAT_DY_THRESH = 6.0
MAX_RMSE_FLAT = 2.5
MIN_R2 = 0.85 

X_POST_MAX = X_SLIT_IN + 100

NBINS_THETA = 40
NBINS_YI = 40

LINEWIDTH_PRE = 0.8
LINEWIDTH_POST = 1.2
ALPHA_PRE = 0.18
ALPHA_POST = 0.70
RAW_LW = 1.0
RAW_ALPHA = 1.0

X_HIST = 350
NBINS_Y = 40

# ================================================================
# ================= SINGLE DENSITY STRIP SETTINGS =================
# ================================================================
X_DENSITY_BAR = 350           # px; slice position
NBINS_DENSITY_BAR = 120       # more bins = smoother strip
DENSITY_STRIP_CMAP = "afmhot" # try: "afmhot", "inferno", "magma", "viridis"
DENSITY_STRIP_WIDTH = 1.8     # width of strip in arbitrary x-units on side panel
DENSITY_STRIP_SMOOTH = 1.2    # Gaussian smoothing in bins; 0 = off
# ================================================================
# ===================== SPEED FILTER SETTINGS =====================
# ================================================================
# We will measure an AVERAGE (mean) speed in a FIXED pre-slit window,
# in mm/s, and keep only tracks with SPEED_MIN <= v_avg <= SPEED_MAX.

USE_T_S_IF_AVAILABLE = True

# Pre-slit window (pixels): choose a region before the slit that is clean/stable
SPEED_X_MIN = X_SLIT_IN - 100   # <-- EDIT (how far before slit to start)
SPEED_X_MAX = X_SLIT_IN - 1    # <-- EDIT (stop just before slit)

# Fixed speed band in mm/s (EDIT THESE TO WHAT YOU WANT)
SPEED_MIN = 8.0   # mm/s
SPEED_MAX = 13.0   # mm/s

# Reject tracks if too many "no-move" steps (optional)
MAX_ZERO_STEP_FRAC = 0.50

# Also keep your old conversion helper (used later in your speed-bin plots)
def px_per_frame_to_mm_per_s(v_px_per_frame: float) -> float:
    return float(v_px_per_frame) * FPS * MM_PER_PX

# Speed-bin analysis (4a): split [SPEED_MIN, SPEED_MAX] into three equal bins
_speed_edges = np.linspace(SPEED_MIN, SPEED_MAX, 4)
SPEED_BINS = [(_speed_edges[i], _speed_edges[i + 1]) for i in range(3)]

# ================================================================
# ============ DEFLECTION ANGLE: "FAR-FIELD" PLATEAU (3) ==========
# ================================================================
PLATEAU_X_START_OFFSET = 0
PLATEAU_X_END_OFFSET   = 50
WINDOW_WIDTH_PX = 50
WINDOW_STEP_PX  = 10
MIN_POINTS_PER_WINDOW = 4

PLATEAU_NEED_CONSEC_WINDOWS = 2
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

# ============================
# THEORETICAL OVERLAYS
# ============================
SLIT_WIDTH_MM_THEORY = SLIT_WIDTH
LAMBDA_MM_THEORY     = 5.193

X_FRESNEL = X_SLIT_IN + 20
NBINS_FRESNEL = 50
NBINS_FRAUNHOFER = NBINS_THETA

# --- Fresnel integrals: SciPy if available, else mpmath fallback ---
try:
    from scipy.special import fresnel as fresnel_cs
    _HAS_FRESNEL = True
except Exception:
    _HAS_FRESNEL = False
    import mpmath as mp

def fresnel_intensity_single_slit(y_mm, a_mm, z_mm, lam_mm):
    y_mm = np.asarray(y_mm, dtype=float)
    a = float(a_mm)
    z = float(z_mm)
    lam = float(lam_mm)

    if z <= 0 or lam <= 0 or a <= 0:
        return np.full_like(y_mm, np.nan, dtype=float)

    pref = np.sqrt(2.0 / (lam * z))
    u1 = pref * (y_mm - a/2.0)
    u2 = pref * (y_mm + a/2.0)

    if _HAS_FRESNEL:
        S1, C1 = fresnel_cs(u1)
        S2, C2 = fresnel_cs(u2)
        dC = C2 - C1
        dS = S2 - S1
        return dC**2 + dS**2
    else:
        I = np.zeros_like(y_mm, dtype=float)
        for i in range(len(y_mm)):
            S1, C1 = mp.fresnel(u1[i])
            S2, C2 = mp.fresnel(u2[i])
            dC = float(C2 - C1)
            dS = float(S2 - S1)
            I[i] = dC*dC + dS*dS
        return I

def fraunhofer_sinc2(theta_deg, a_mm, lam_mm):
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
    beta = (np.pi * float(a_mm) / float(lam_mm)) * np.sin(theta)
    I = np.ones_like(beta, dtype=float)
    m = np.abs(beta) > 1e-12
    I[m] = (np.sin(beta[m]) / beta[m])**2
    return I

def plot_yi_speed_hexbin_only(yi_mm, sp_mm_s, gridsize=22, log_counts=True, mincnt=1,
                             ymin=None, ymax=None):
    yi_mm = np.asarray(yi_mm, float)
    sp_mm_s = np.asarray(sp_mm_s, float)
    m = np.isfinite(yi_mm) & np.isfinite(sp_mm_s)
    yi_mm, sp_mm_s = yi_mm[m], sp_mm_s[m]

    fig, ax = plt.subplots(figsize=(6.6, 5.2))

    hb = ax.hexbin(
        yi_mm, sp_mm_s,
        gridsize=gridsize,
        mincnt=mincnt,
        bins="log" if log_counts else None,
        cmap = 'rainbow'
    )

    cbar = fig.colorbar(hb, ax=ax, pad=0.02)
    cbar.set_label("log10(count)" if log_counts else "Counts")

    ax.axvline(0, color="k", lw=1.0, alpha=0.5)
    ax.set_xlabel(r"Impact parameter $y_i$ (mm)")
    ax.set_ylabel("Droplet mean pre-slit speed (mm/s)")
    ax.set_title("Impact parameter vs  Droplet mean speed (density)")
    ax.grid(True, alpha=0.25)

    # --- Force y-axis limits if requested ---
    if ymin is None:
        ymin = float(np.nanmin(sp_mm_s)) if sp_mm_s.size else 0.0
    if ymax is None:
        ymax = float(np.nanmax(sp_mm_s)) if sp_mm_s.size else 1.0
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "Impact parameter vs pre-slit speed hexbin.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    
    plt.show()
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

    for c in ["frame", "x_px", "y_px"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "t_s" in df.columns:
        df["t_s"] = pd.to_numeric(df["t_s"], errors="coerce")

    df = df.dropna(subset=["frame", "x_px", "y_px"])
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

# ================================================================
# ========================= SPEED HELPERS =========================
# ================================================================
def compute_track_speed_mean_preslit(d: pd.DataFrame,
                                    x_min_px: float,
                                    x_max_px: float,
                                    use_t_s_if_available: bool,
                                    fps: float,
                                    mm_per_px: float,
                                    max_zero_step_frac=None):
    """
    Compute MEAN speed (mm/s) using step distances inside [x_min_px, x_max_px] before the slit.
    Returns (v_mean_mm_s, status).
    """
    g = d.sort_values("frame")
    x = g["x_px"].to_numpy(dtype=float)
    y = g["y_px"].to_numpy(dtype=float)

    m = np.ones(len(g), dtype=bool)
    if x_min_px is not None:
        m &= (x >= x_min_px)
    if x_max_px is not None:
        m &= (x <= x_max_px)

    if np.count_nonzero(m) < 3:
        return None, "no_points_in_speed_window"

    xw = x[m]
    yw = y[m]

    dt = None
    if use_t_s_if_available and ("t_s" in g.columns):
        tw = g.loc[m, "t_s"].to_numpy(dtype=float)
        if np.isfinite(tw).all() and np.ptp(tw) > 0:
            dt = np.diff(tw)

    if dt is None:
        fr = g.loc[m, "frame"].to_numpy(dtype=float)
        dt = np.diff(fr)
        if fps is None or fps <= 0:
            return None, "fps_missing"
        dt = dt / float(fps)

    dist_px = np.hypot(np.diff(xw), np.diff(yw))
    dist_mm = dist_px * float(mm_per_px)

    good = dt > 0
    if np.count_nonzero(good) < 1:
        return None, "bad_dt"

    v_steps = dist_mm[good] / dt[good]
    if v_steps.size < 2:
        return None, "not_enough_steps"

    if max_zero_step_frac is not None:
        zero_frac = np.mean(dist_mm[good] <= 1e-12)
        if zero_frac > max_zero_step_frac:
            return None, "too_many_zero_steps"

    return float(np.nanmean(v_steps)), "ok"

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
# ============================ FILTERS ============================
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

def plot_raw_trajectories_coloured_by_speed_with_cross_filter(
    df, speed_by_key, group_cols, mm_per_px,
    x_slit_in_px, y_slit_center_px,
    title="ALL trajectories (raw) coloured by mean pre-slit speed (5 bins)\n(cross-x-slit only)"
):
    """
    Same as your original plot, but applies the 'no_cross_xslit' filter:
    only trajectories that cross x = x_slit_in_px are plotted.
    Trajectories are coloured by per-track mean pre-slit speed using 5 bins (quintiles).
    """

    # --- First: get keys that actually CROSS x_slit_in_px ---
    keys_cross = []
    v_list = []

    for key, d in df.groupby(group_cols, sort=False):
        x = d["x_px"].to_numpy(dtype=float)
        y = d["y_px"].to_numpy(dtype=float)

        yi_abs = y_at_x_first_crossing(x, y, x_slit_in_px)
        if yi_abs is None:
            continue  # <-- this is the no_cross_xslit filter

        v = speed_by_key.get(key, None)
        if v is None or not np.isfinite(v):
            continue  # we can't colour it meaningfully, so skip (you can change this if you want)

        keys_cross.append(key)
        v_list.append(float(v))

    v_arr = np.asarray(v_list, dtype=float)
    if v_arr.size < 5:
        print("Not enough cross-x-slit tracks with valid speeds to make 5 bins.")
        return

    # --- 5 bins: quintiles (equal counts) ---
    edges = np.quantile(v_arr, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    edges = np.unique(edges)
    if edges.size < 6:
        edges = np.linspace(np.min(v_arr), np.max(v_arr), 6)

    cmap = plt.get_cmap("viridis", len(edges) - 1)
    norm = mpl.colors.BoundaryNorm(edges, cmap.N)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 6))

    n_plotted = 0
    n_skipped_no_cross = 0
    n_skipped_no_speed = 0

    for key, d in df.groupby(group_cols, sort=False):
        x = d["x_px"].to_numpy(dtype=float)
        y = d["y_px"].to_numpy(dtype=float)

        # Apply the no_cross_xslit filter
        yi_abs = y_at_x_first_crossing(x, y, x_slit_in_px)
        if yi_abs is None:
            n_skipped_no_cross += 1
            continue

        v = speed_by_key.get(key, None)
        if v is None or not np.isfinite(v):
            n_skipped_no_speed += 1
            continue

        ax.plot(x * mm_per_px, y * mm_per_px, color=cmap(norm(v)), lw=0.9, alpha=0.95)
        n_plotted += 1

    # Geometry guides
    ax.axvline(x_slit_in_px * mm_per_px, color="k", ls="--", lw=1.2, alpha=0.8, label="slit x")
    ax.axhline(y_slit_center_px * mm_per_px, color="k", ls=":", lw=1.0, alpha=0.6, label="slit centre")

    ax.invert_yaxis()
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"{title}\n(plotted N={n_plotted}; skipped no-cross={n_skipped_no_cross}, no-speed={n_skipped_no_speed})")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    # Colorbar with bin edges
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Mean pre-slit speed (mm/s)")
    cbar.set_ticks(edges)
    cbar.ax.set_yticklabels([f"{e:.2f}" for e in edges])

    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "All trajectories raw coloured by mean pre-slit speed cross-x-slit only.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    
def plot_kept_trajectories_with_density_strip(
    kept,
    mm_per_px,
    x_slit_in_px,
    y_split_px,
    x_post_max_px,
    x_density_px,
    nbins_density=120,
    cmap="afmhot",
    strip_width=1.8,
    smooth_sigma_bins=1.2,
    title=None,
    show_density_strip=True
):
    """
    Main trajectory plot + optional vertical density strip on the right.
    The strip shows the y-density at x = x_density_px.

    Slit drawing is controlled by global settings at the top of the script.
    """

    if show_density_strip:
        fig = plt.figure(figsize=(8.8, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[5.4, 0.75], wspace=0.06)
        ax = fig.add_subplot(gs[0, 0])
        ax_strip = fig.add_subplot(gs[0, 1], sharey=ax)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax_strip = None

    # ------------------------------------------------------------
    # Main trajectory plot
    # ------------------------------------------------------------
    y_samples_mm = []

    for x, y, yi, theta, col, r2_like, sp, theta_status in kept:
        pre = x <= x_slit_in_px
        if np.any(pre):
            ax.plot(
                x[pre] * mm_per_px,
                y[pre] * mm_per_px,
                color="0.3",
                lw=LINEWIDTH_PRE,
                alpha=ALPHA_PRE
            )

        post = (x >= x_slit_in_px) & (x <= x_post_max_px)
        if np.any(post):
            ax.plot(
                x[post] * mm_per_px,
                y[post] * mm_per_px,
                color=col,
                lw=LINEWIDTH_POST,
                alpha=ALPHA_POST
            )

        if show_density_strip:
            y_at_bar = y_at_x_first_crossing(x, y, x_density_px)
            if y_at_bar is not None:
                y_samples_mm.append(y_at_bar * mm_per_px)

    y_samples_mm = np.asarray(y_samples_mm, dtype=float)

    # ------------------------------------------------------------
    # Guide lines
    # ------------------------------------------------------------
    ax.axvline(
        x_slit_in_px * mm_per_px,
        color="k", ls="--", lw=1.2, alpha=0.8, label="Slit exit x"
    )
    ax.axhline(
        y_split_px * mm_per_px,
        color="gold", ls="--", lw=1.2, alpha=0.9, label="Launcher y"
    )

    if show_density_strip:
        ax.axvline(
            x_density_px * mm_per_px,
            color="red", ls="-", lw=2.0, alpha=0.8,
            label=f"Density slice x = {x_density_px * mm_per_px:.1f} mm"
        )

    # ------------------------------------------------------------
    # Axes formatting
    # ------------------------------------------------------------
    ax.invert_yaxis()
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_xlim(0,41)
    ax.set_ylim(55, 15)
    ax.set_title(title if title is not None else f"Droplet Trajectories (kept N={len(kept)}), s={SLIT_WIDTH:g}mm")
    style_axes(ax)

    # ------------------------------------------------------------
    # Slit blocks
    # ------------------------------------------------------------
    if SHOW_SLIT_BLOCKS:
        x_slit_mm = x_slit_in_px * mm_per_px
        y_slit_centre_mm = Y_SLIT_CENTER_PX * mm_per_px
        slit_width_mm = SLIT_WIDTH_PX * mm_per_px
        plate_thickness_mm = PLATE_THICKNESS_PX * mm_per_px
        slit_x_offset_mm = SLIT_X_OFFSET_PX * mm_per_px

        y0, y1 = ax.get_ylim()

        y_top = min(y0, y1)
        y_bottom = max(y0, y1)
        
        slit_top_mm = y_slit_centre_mm - 0.5 * slit_width_mm
        slit_bottom_mm = y_slit_centre_mm + 0.5 * slit_width_mm
        
        x_left_mm = x_slit_mm - slit_x_offset_mm
        x_right_mm = x_left_mm + plate_thickness_mm
        
        # top block
        ax.add_patch(Rectangle(
            (x_left_mm, y_top),
            plate_thickness_mm,
            slit_top_mm - y_top,
            facecolor=SLIT_FACECOLOR,
            edgecolor="none",
            alpha=SLIT_ALPHA,
            zorder=0.1
        ))
        
        # bottom block
        ax.add_patch(Rectangle(
            (x_left_mm, slit_bottom_mm),
            plate_thickness_mm,
            y_bottom - slit_bottom_mm,
            facecolor=SLIT_FACECOLOR,
            edgecolor="none",
            alpha=SLIT_ALPHA,
            zorder=0.1
        ))

       # Outline the barrier
    ax.plot([x_left_mm, x_left_mm], [y_top, slit_top_mm],
            color=SLIT_EDGECOLOR, lw=SLIT_EDGEWIDTH, zorder=2)
    ax.plot([x_right_mm, x_right_mm], [y_top, slit_top_mm],
            color=SLIT_EDGECOLOR, lw=SLIT_EDGEWIDTH, zorder=2)
     
    ax.plot([x_left_mm, x_left_mm], [slit_bottom_mm, y_bottom],
            color=SLIT_EDGECOLOR, lw=SLIT_EDGEWIDTH, zorder=2)
    ax.plot([x_right_mm, x_right_mm], [slit_bottom_mm, y_bottom],
            color=SLIT_EDGECOLOR, lw=SLIT_EDGEWIDTH, zorder=2)
     
    ax.plot([x_left_mm, x_right_mm], [slit_top_mm, slit_top_mm],
            color=SLIT_EDGECOLOR, lw=SLIT_EDGEWIDTH, zorder=2)
    ax.plot([x_left_mm, x_right_mm], [slit_bottom_mm, slit_bottom_mm],
            color=SLIT_EDGECOLOR, lw=SLIT_EDGEWIDTH, zorder=2)

    ax.legend(frameon=False, loc="upper left")

    # ------------------------------------------------------------
    # Density strip
    # ------------------------------------------------------------
    if show_density_strip:
        ax_strip.set_ylim(ax.get_ylim())

        if y_samples_mm.size < 2:
            ax_strip.text(
                0.5, 0.5, "Not enough\ncrossings",
                ha="center", va="center",
                transform=ax_strip.transAxes
            )
            ax_strip.set_axis_off()
        else:
            y0, y1 = ax.get_ylim()
            y_low = min(y0, y1)
            y_high = max(y0, y1)

            density, edges = np.histogram(
                y_samples_mm,
                bins=nbins_density,
                range=(y_low, y_high),
                density=True
            )

            if smooth_sigma_bins is not None and smooth_sigma_bins > 0:
                try:
                    from scipy.ndimage import gaussian_filter1d
                    density = gaussian_filter1d(density, sigma=smooth_sigma_bins, mode="nearest")
                except Exception:
                    pass

            strip_img = np.tile(density[:, np.newaxis], (1, 20))

            im = ax_strip.imshow(
                strip_img,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                extent=[0, strip_width, y_low, y_high]
            )

            ax_strip.set_xlim(0, strip_width)
            ax_strip.set_xticks([])
            ax_strip.grid(False)

            plt.setp(ax_strip.get_yticklabels(), visible=False)
            ax_strip.tick_params(axis="y", length=0)

            cbar = fig.colorbar(im, ax=ax_strip, pad=0.03)
            cbar.set_label("Density")

    fig.tight_layout()

    out_path = os.path.join(OUTPUT_FOLDER, "Kept trajectories with density strip and slit.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
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
    ax.axhline(Y_SPLIT * MM_PER_PX, color="gold", ls="--", lw=1.2, alpha=0.9, label="launch split")
    ax.invert_yaxis()
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"ALL trajectories (raw, N={n_tracks_total})")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "All trajectories raw.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================================
    # PASS 1: compute MEAN pre-slit speeds (mm/s) using the NEW method
    # ============================================================
    speed_by_key = {}
    speed_reject = {}

    for key, d in df.groupby(group_cols, sort=False):
        v_mean, st = compute_track_speed_mean_preslit(
            d,
            x_min_px=SPEED_X_MIN,
            x_max_px=SPEED_X_MAX,
            use_t_s_if_available=USE_T_S_IF_AVAILABLE,
            fps=FPS,
            mm_per_px=MM_PER_PX,
            max_zero_step_frac=MAX_ZERO_STEP_FRAC
        )
        if st != "ok":
            speed_reject[st] = speed_reject.get(st, 0) + 1
            speed_by_key[key] = None
        else:
            speed_by_key[key] = v_mean

    all_speeds = np.array([v for v in speed_by_key.values() if v is not None and np.isfinite(v)], dtype=float)
    if all_speeds.size == 0:
        print("No valid speeds computed — check t_s / FPS / speed window.")
        return

    vmin = float(SPEED_MIN)
    vmax = float(SPEED_MAX)

    print("\n================ SPEED SUMMARY (NEW) ================")
    print("Speed units: mm/s")
    print(f"Speed computed in pre-slit x-window (px): [{SPEED_X_MIN}, {SPEED_X_MAX}]")
    print("Speed statistic: mean (average step-speed)")
    print(f"Computed speeds for {len(all_speeds)} tracks.")
    print(f"Speed percentiles (10/50/90): "
          f"{np.percentile(all_speeds,10):.3f}, {np.percentile(all_speeds,50):.3f}, {np.percentile(all_speeds,90):.3f} mm/s")
    print(f"Speed filter: keep [{vmin:.3f}, {vmax:.3f}] mm/s")
    if speed_reject:
        print("Speed computation rejects:")
        for k in sorted(speed_reject.keys()):
            print(f"  {k:>28s} : {speed_reject[k]}")
    print("====================================================\n")

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
    out_path = os.path.join(OUTPUT_FOLDER, "Trajectory speed distribution.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    # ---- NEW: Raw trajectories coloured by speed (5 bins) ----
    plot_raw_trajectories_coloured_by_speed_with_cross_filter(
        df=df,
        speed_by_key=speed_by_key,
        group_cols=group_cols,
        mm_per_px=MM_PER_PX,
        x_slit_in_px=X_SLIT_IN,
        y_slit_center_px=Y_SLIT_CENTER_PX,
        title="ALL trajectories (raw) coloured by mean pre-slit speed (5 bins)\n(cross-x-slit only)"
    )
    # ============================================================
    # NEW: Impact parameter vs speed BEFORE speed band is applied
    #      (cross-x-slit only, but NO SPEED_MIN/MAX filtering)
    # ============================================================
    yi_pre_mm = []
    sp_pre = []
    
    for key, d in df.groupby(group_cols, sort=False):
        x = d["x_px"].to_numpy(dtype=float)
        y = d["y_px"].to_numpy(dtype=float)
    
        # must cross the slit x to define yi at the slit
        yi_abs = y_at_x_first_crossing(x, y, X_SLIT_IN)
        if yi_abs is None:
            continue  # this is the no_cross_xslit filter
    
        # use the already-computed mean pre-slit speed from PASS 1
        v = speed_by_key.get(key, None)
        if v is None or (not np.isfinite(v)):
            continue  # skip if speed couldn't be computed
    
        yi_pre_mm.append((yi_abs - Y_SLIT_CENTER_PX) * MM_PER_PX)
        sp_pre.append(v)
    
    yi_pre_mm = np.asarray(yi_pre_mm, dtype=float)
    sp_pre = np.asarray(sp_pre, dtype=float)
    
    print(f"\nPre-speed-cut dataset (cross-slit + valid speed): N={len(sp_pre)}")
    if len(sp_pre) > 0:
        print(f"Speed range (pre-cut): {np.min(sp_pre):.3f} to {np.max(sp_pre):.3f} mm/s")
    
    # Now plot using your hexbin-only function (no overlays)
    plot_yi_speed_hexbin_only(
        yi_mm=yi_pre_mm,
        sp_mm_s=sp_pre,
        gridsize=20,      # adjust tile size
        log_counts=False,  # boosts contrast
        ymin=0            # optional; remove if you want auto scaling
    )
    # ============================================================
    # PASS 2: Filter + compute theta
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
        yi = yi_abs - Y_SLIT_CENTER_PX

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
    # Plot: Impact parameter vs speed (hexbin + median/IQR)
    # ============================================================
    #plot_yi_speed_hexbin_only(yi_arr_mm, sp_arr, gridsize=22, log_counts=False, ymin=0)
    # ============================================================
    # Histogram of y positions at x=X_HIST (kept) — absolute y in mm
    # ============================================================
    y_hist_kept_px = []
    for x, y, yi, theta, col, r2_like, sp, theta_status in kept:
        y_at_hist = y_at_x_first_crossing(x, y, X_HIST)
        if y_at_hist is not None:
            y_hist_kept_px.append(y_at_hist)

    y_hist_kept_px = np.array(y_hist_kept_px, dtype=float)
    y0_mm = Y_SLIT_CENTER_PX * MM_PER_PX
    y_hist_kept_mm_rel = (y_hist_kept_px * MM_PER_PX) - y0_mm

    fig, ax = plt.subplots(figsize=(6, 5))
    if len(y_hist_kept_mm_rel) == 0:
        ax.text(0.5, 0.5, f"No KEPT tracks cross x={X_HIST * MM_PER_PX:.2f} mm", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(y_hist_kept_mm_rel, bins=NBINS_Y, alpha=0.85)
        ax.axvline(0, color="k", ls="--", lw=1.2, alpha=0.8, label="slit centre")
        ax.set_xlabel(f"Centred y positions at x = {X_HIST * MM_PER_PX:.2f} mm (mm)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Histogram of y positions in the Fresnel case (x={X_HIST * MM_PER_PX:.2f} mm)\n"
            f"(N={len(y_hist_kept_mm_rel)} trajectories)"
        )
        style_axes(ax)
        ax.legend(frameon=False)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "Histogram of y positions at Fresnel plane.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================================
    # 4b: yi histogram at slit (mm)
    # ============================================================
    if DO_YI_HIST:
        fig, ax = plt.subplots(figsize=(6, 4.8))
        ax.hist(yi_arr_mm, bins=NBINS_YI, alpha=0.85)
        ax.axvline(0, color="k", ls="--", lw=1.2, alpha=0.8, label="slit centre (yi=0)")
        ax.set_xlabel("Impact parameter $y_i$ (mm; y_at_slit - slit_center)")
        ax.set_ylabel("Count")
        ax.set_title(f"Impact parameter distribution at slit (kept N={N})")
        style_axes(ax)
        ax.legend(frameon=False)
        fig.tight_layout()
        out_path = os.path.join(OUTPUT_FOLDER, "Impact parameter histogram at slit.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.show()

        # ============================================================
        # ============================================================
    # Plot 1: Trajectories with single density strip at chosen x
    # ============================================================
    plot_kept_trajectories_with_density_strip(
        kept,
        MM_PER_PX,
        X_SLIT_IN,
        Y_SPLIT,
        X_POST_MAX,
        X_DENSITY_BAR,
        nbins_density=120,
        cmap="afmhot",
        strip_width=1.8,
        smooth_sigma_bins=1.2,
        title=None,
        show_density_strip=True
    )

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
    out_path = os.path.join(OUTPUT_FOLDER, "Post-slit fan slit-aligned.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================================
    # Plot 3: theta vs yi
    # ============================================================
    cols = np.array([k[4] for k in kept])
    fig, ax = plt.subplots(figsize=(6, 5))
    
    for c in [COLOR_A, COLOR_B]:
        m = cols == c
        ax.scatter(th_arr[m], yi_arr_mm[m], s=35, alpha=0.9, label=c, color=c)
    
    ax.axvline(0, color="k", lw=1.0, alpha=0.6)
    ax.axhline(0, color="k", lw=1.0, alpha=0.6)
    ax.set_ylabel("Impact parameter $y_i$ (mm, relative to slit centre)")
    ax.set_xlabel("Deflection angle $\\theta$ (deg, from horizontal)")
    ax.set_title(f"Deflection Angle vs Impact Parameter (N={N})")
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    
    out_path = os.path.join(OUTPUT_FOLDER, "Impact parameter vs deflection angle.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    
    plt.show()

    # ============================================================
    # Plot 4: theta histogram
    # ============================================================
    fig, ax = plt.subplots(figsize=(6, 5))
    theta_bins = np.linspace(-90, 90, NBINS_THETA + 1)
    ax.hist(th_arr, bins=theta_bins, density=True, histtype="stepfilled", alpha=0.55)
    ax.set_xlim(-90, 90)
    ax.set_xlabel("$\\theta$ (deg, from horizontal)")
    ax.set_ylabel("$P(\\theta)$")
    ax.set_title(f"Deflection angle distribution for Fraunhofer case (kept N={N})")
    style_axes(ax)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "Deflection angle distribution.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================================
    # 4b: Reweighted theta histogram
    # ============================================================
    if DO_REWEIGHTED_THETA_HIST:
        yi_edges = np.linspace(np.min(yi_arr_mm), np.max(yi_arr_mm), REWEIGHT_YI_BINS + 1)
        yi_bin = np.digitize(yi_arr_mm, yi_edges) - 1
        yi_bin = np.clip(yi_bin, 0, REWEIGHT_YI_BINS - 1)
        counts = np.bincount(yi_bin, minlength=REWEIGHT_YI_BINS).astype(float)
        counts[counts < 1] = 1.0
        weights = 1.0 / counts[yi_bin]
        weights = np.clip(weights, 0, np.percentile(weights, 90))

        centres_w, dens_w, edges_w = weighted_hist_density(th_arr, bins=theta_bins, weights=weights)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.step(centres_w, dens_w, where="mid", lw=2.0, alpha=0.9, label="reweighted by 1/count(yi-bin)")
        ax.set_xlim(-90, 90)
        ax.set_xlabel("$\\theta$ (deg, from horizontal)")
        ax.set_ylabel("$P(\\theta)$ (reweighted)")
        ax.set_title(f"Reweighted $P(\\theta)$ (compensate non-uniform $y_i$)\n(kept N={N})")
        style_axes(ax)
        ax.legend(frameon=False)
        fig.tight_layout()
        out_path = os.path.join(OUTPUT_FOLDER, "Reweighted deflection angle distribution.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.show()

    # ============================================================
    # 10–12 px/frame-equivalent band -> convert to mm/s band (your original plot)
    # ============================================================
    bins_theta = np.linspace(-90, 90, NBINS_THETA + 1)
    centres3 = 0.5 * (bins_theta[:-1] + bins_theta[1:])

    v_lo = px_per_frame_to_mm_per_s(10.0)
    v_hi = px_per_frame_to_mm_per_s(12.0)
    m = (sp_arr >= v_lo) & (sp_arr < v_hi)
    theta_sub = th_arr[m]
    print(f"10–12 px/frame-equivalent band ({v_lo:.2f}–{v_hi:.2f} mm/s): N={theta_sub.size}")

    hist3, _ = np.histogram(theta_sub, bins=bins_theta, density=True)

    def sinc_sq(beta):
        out = np.ones_like(beta, dtype=float)
        mask = np.abs(beta) > 1e-12
        out[mask] = (np.sin(beta[mask]) / beta[mask])**2
        return out

    theta0_deg = 30.0
    k = np.pi / np.sin(np.deg2rad(theta0_deg))
    beta = k * np.sin(np.deg2rad(centres3))
    model = sinc_sq(beta)

    model_scaled = model / (model.max() if model.max() > 0 else 1.0) * (hist3.max() if hist3.max() > 0 else 1.0)

    fig = plt.figure(figsize=(10.5, 4.8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3.2, 1.0], wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(theta_sub, bins=bins_theta, density=True, alpha=0.45,
            label=f"Data (speed {v_lo:.2f}–{v_hi:.2f} mm/s)")
    ax.plot(centres3, model_scaled, lw=2.2,
            label=rf"Single-slit model ($\theta_0 \approx {theta0_deg:.0f}^\circ$ first zero)")
    ax.set_xlim(-90, 90)
    ax.set_xlabel(r"$\theta$ (deg, from horizontal)")
    ax.set_ylabel(r"$P(\theta)$")
    ax.set_title(f"Deflection angles + single-slit overlay (N={theta_sub.size})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax2 = fig.add_subplot(gs[0, 1])
    I = model / (model.max() if model.max() > 0 else 1.0)
    H = 220
    img = np.tile(I[np.newaxis, :], (H, 1))
    ax2.imshow(img, aspect="auto", origin="lower",
               extent=[-90, 90, 0, 1], cmap="gray_r")
    ax2.set_xlim(-90, 90)
    ax2.set_yticks([])
    ax2.set_xlabel(r"$\theta$")
    ax2.set_title("Fringe pattern")
    ax2.axvline(0, color="k", lw=0.8, alpha=0.6)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "Deflection angles with single-slit overlay.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================================
    # 4a: Theta histograms per speed bin (mm/s)
    # ============================================================
    fig, ax = plt.subplots(figsize=(7, 5))
    for (a, b) in SPEED_BINS:
        m = (sp_arr >= a) & (sp_arr < b)
        if np.count_nonzero(m) < 5:
            continue
        ax.hist(th_arr[m], bins=bins_theta, density=True, histtype="step", lw=2.0,
                label=f"{a:.2f}–{b:.2f} mm/s (N={np.count_nonzero(m)})")

    ax.set_xlim(-90, 90)
    ax.set_xlabel("$\\theta$ (deg, from horizontal)")
    ax.set_ylabel("$P(\\theta)$")
    ax.set_title("Deflection angle distributions by speed bin (4a)")
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "Deflection angle distributions by speed bin.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================================
    # 4c diagnostics
    # ============================================================
    theta_statuses = [k[7] for k in kept]
    n_plateau = sum(s == "plateau" for s in theta_statuses)
    n_fallback = sum(s == "fallback_fixed_window" for s in theta_statuses)

    print("\n================ 4c DIAGNOSTIC (code-based) ================")
    print(f"  plateau theta measured: {n_plateau}")
    print(f"  fallback (near-slit window): {n_fallback}")
    print("============================================================\n")

    # ============================================================
    # 1) FRESNEL: position histogram at x = X_FRESNEL
    # ============================================================
    y_fresnel_rel_mm = []
    for x, y, yi, theta, col, r2_like, sp, theta_status in kept:
        y_at_plane = y_at_x_first_crossing(x, y, X_FRESNEL)
        if y_at_plane is None:
            continue
        y_rel_mm = (y_at_plane - Y_SLIT_CENTER_PX) * MM_PER_PX
        y_fresnel_rel_mm.append(y_rel_mm)

    y_fresnel_rel_mm = np.array(y_fresnel_rel_mm, dtype=float)
    z_mm = (X_FRESNEL - X_SLIT_IN) * MM_PER_PX

    fig, ax = plt.subplots(figsize=(6.5, 5))
    if y_fresnel_rel_mm.size < 5:
        ax.text(0.5, 0.5, "Not enough tracks cross the Fresnel plane.\nIncrease X_FRESNEL or relax filters.",
                ha="center", va="center")
        ax.set_axis_off()
    else:
        counts, edges, _ = ax.hist(y_fresnel_rel_mm, bins=NBINS_FRESNEL, density=True,
                                   alpha=0.55, label=f"Data (N={y_fresnel_rel_mm.size})")
        centres_y = 0.5 * (edges[:-1] + edges[1:])

        I_th = fresnel_intensity_single_slit(centres_y, SLIT_WIDTH_MM_THEORY, z_mm, LAMBDA_MM_THEORY)
        I_th = I_th / (np.nanmax(I_th) if np.nanmax(I_th) > 0 else 1.0)
        hist_peak = np.max(counts) if len(counts) else 1.0
        I_plot = I_th * hist_peak

        ax.plot(centres_y, I_plot, lw=2.2, color='black', label="Fresnel theory")
        ax.axvline(0, color="k", ls="--", lw=1.2, alpha=0.8, label="slit centre")
        ax.set_xlabel(r"$y$ position relative to slit centre (mm)")
        ax.set_ylabel(r"Probability density")
        ax.set_title(f"Fresnel (near-field) pattern at x = {X_FRESNEL*MM_PER_PX:.2f} mm\n"
                     f"(z = {z_mm:.2f} mm after slit)")
        style_axes(ax)
        ax.legend(frameon=False)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "Fresnel pattern with theory.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================================
    # 2) FRAUNHOFER: angle histogram with shifted sinc^2 theory
    # ============================================================
    th_ff = th_arr[np.isfinite(th_arr)]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    if th_ff.size < 5:
        ax.text(0.5, 0.5, "Not enough angles for Fraunhofer plot.", ha="center", va="center")
        ax.set_axis_off()
    else:
        theta_bins = np.linspace(-90, 90, NBINS_FRAUNHOFER + 1)
        counts, edges, _ = ax.hist(th_ff, bins=theta_bins, density=True, alpha=0.55,
                                   label=f"Data (N={th_ff.size})")
        centres_t = 0.5 * (edges[:-1] + edges[1:])
        theta_shift = float(np.median(th_ff))

        I_th = fraunhofer_sinc2(centres_t - theta_shift, SLIT_WIDTH_MM_THEORY, LAMBDA_MM_THEORY)
        I_th = I_th / (np.max(I_th) if np.max(I_th) > 0 else 1.0)

        hist_peak = np.max(counts) if len(counts) else 1.0
        I_plot = I_th * hist_peak

        ax.plot(centres_t, I_plot, lw=2.2, color='black', label="Fraunhofer theory")
        ax.axvline(theta_shift, color="k", ls="--", lw=1.2, alpha=0.8, label=r"data centre $\theta_0$")
        ax.set_xlim(-90, 90)
        ax.set_xlabel(r"$\theta$ (deg)")
        ax.set_ylabel(r"$P(\theta)$ / scaled intensity")
        ax.set_title("Fraunhofer (far-field) angular distribution")
        style_axes(ax)
        ax.legend(frameon=False)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "Fraunhofer angular distribution with theory.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    # ============================================================
    # Theoretical Fraunhofer overlays for multiple slit widths
    # ============================================================
    slit_widths_mm = [10, 12, 17, 22]
    theta_plot = np.linspace(-90, 90, 1000)
    
    fig, ax = plt.subplots(figsize=(4.2, 5.2))
    for a_mm in slit_widths_mm:
        I_th = fraunhofer_sinc2(theta_plot, a_mm, LAMBDA_MM_THEORY)
        I_th = I_th / np.max(I_th)
        ax.plot(theta_plot, I_th, lw=2.0, label=fr"{a_mm} mm")
    
    ax.axvline(0, color="k", ls="--", lw=1.2, alpha=0.8)
    ax.set_xlim(-75, 75)
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel("Normalised intensity")
    ax.set_title("Theoretical Fraunhofer plots for different slit widths")
    style_axes(ax)
    ax.legend(title="Slit width", frameon=False)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "Theoretical Fraunhofer plots for different slit widths.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    # ============================================================
    # Reverse engineer Faraday wavelength (your grid search block)
    # ============================================================
    a_mm = SLIT_WIDTH
    theta_data = th_arr[np.isfinite(th_arr)]

    NBINS = 40
    theta_bins = np.linspace(-90, 90, NBINS + 1)
    centres = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    dtheta = centres[1] - centres[0]

    hist, _ = np.histogram(theta_data, bins=theta_bins, density=True)

    def sinc2(beta):
        out = np.ones_like(beta, dtype=float)
        m = np.abs(beta) > 1e-12
        out[m] = (np.sin(beta[m]) / beta[m])**2
        return out

    def gauss_smooth(y, sigma_deg):
        if sigma_deg <= 0:
            return y
        sigma_bins = sigma_deg / dtheta
        r = int(np.ceil(4*sigma_bins))
        k = np.arange(-r, r+1)
        g = np.exp(-0.5*(k/sigma_bins)**2)
        g /= g.sum()
        return np.convolve(y, g, mode="same")

    def model_hist(lam_mm, theta0_deg, sigma_deg):
        th = np.deg2rad(centres - theta0_deg)
        beta = (np.pi * a_mm / lam_mm) * np.sin(th)
        I = sinc2(beta)
        I = gauss_smooth(I, sigma_deg)
        scale = (hist @ I) / (I @ I) if (I @ I) > 0 else 1.0
        return scale * I

    def sse(lam_mm, theta0_deg, sigma_deg):
        m = model_hist(lam_mm, theta0_deg, sigma_deg)
        return float(np.mean((hist - m)**2))

    lam_grid    = np.linspace(1.5, 10.0, 250)
    theta0_grid = np.linspace(-15, 15, 121)
    sigma_grid  = np.linspace(0.0, 10.0, 81)

    # best = (np.inf, None, None, None)
    # for lam in lam_grid:
    #     for theta0 in theta0_grid:
    #         for sig in sigma_grid:
    #             val = sse(lam, theta0, sig)
    #             if val < best[0]:
    #                 best = (val, lam, theta0, sig)

    # best_sse, best_lam, best_theta0, best_sigma = best
    # print(f"Best-fit lambda ≈ {best_lam:.3f} mm")
    # print(f"Best-fit theta0 ≈ {best_theta0:+.2f} deg (offset)")
    # print(f"Best-fit sigma  ≈ {best_sigma:.2f} deg (blur)")
    # print(f"SSE = {best_sse:.6e}")

    # fit_curve = model_hist(best_lam, best_theta0, best_sigma)

    # fig, ax = plt.subplots(figsize=(6.5, 5))
    # ax.hist(theta_data, bins=theta_bins, density=True, alpha=0.55, label="Data")
    # ax.plot(centres, fit_curve, lw=2.4,
    #         label=rf"Fit sinc$^2$ (a={a_mm:.1f} mm, $\lambda$={best_lam:.2f} mm, "
    #               rf"$\theta_0$={best_theta0:+.1f}°, $\sigma$={best_sigma:.1f}°)")
    # ax.axvline(best_theta0, color="k", ls="--", lw=1.1, alpha=0.7)
    # ax.set_xlim(-90, 90)
    # ax.set_xlabel(r"$\theta$ (deg)")
    # ax.set_ylabel(r"$P(\theta)$")
    # ax.set_title("Reverse-engineered Faraday wavelength from Fraunhofer P(θ)")
    # ax.grid(True, alpha=0.25)
    # ax.legend(frameon=False)
    # fig.tight_layout()
    # plt.show()
    
    

if __name__ == "__main__":
    main()
