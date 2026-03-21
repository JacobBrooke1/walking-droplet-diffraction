import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import time
from collections import deque
from math import atan2, sqrt
from vmbpy import VmbSystem, PixelFormat  # NOTE: PixelFormat added
from datetime import datetime

# ================================================================
# ======================= USER SETTINGS ===========================
# ================================================================

# Folder to save trajectory + plots
OUTPUT_FOLDER = r"C:\Users\ck23368\OneDrive - University of Bristol\3rd Year\Research Project\Tracking plots"

# If you know your actual acquisition frame rate, set it here (Hz).
# Used to compute t_s = frame / FRAME_RATE_HZ in the long-format CSV.
FRAME_RATE_HZ = None  # e.g. 32.8   (set None to save t_s as NaN)

# Region of interest as fractions of full image (x0, y0, x1, y1)
ROI_FRAC = (0.0, 0.15, 0.5, 0.5)

# --- Droplet appearance ---
DROPLET_IS_DARK = False   # False = droplet bright on dark
THRESH_VAL = 10           # lower for faint droplets

# ---------------- Slit / measurement geometry ----------------
X_SLIT_IN = 300      # where droplet reaches slit entrance (or your reference line)
X_ANGLE_START = 300
X_ANGLE_END   = 310
# -------------------------------------------------------------

# Blob size/shape limits (in ROI pixels)
MAX_BLOB_WIDTH  = 25
MAX_BLOB_HEIGHT = 25
MAX_ASPECT_RATIO = 2.0
MIN_CIRCULARITY = 0.6

# Detection parameters (area in pixels)
MIN_AREA = 5
MAX_AREA = 60

# Tracking continuity
MAX_JUMP_PIXELS = 40
MAX_MISSES = 5
MIN_TRACK_LENGTH = 25

# Background subtraction inside ROI
BG_INIT_FRAMES = 40
BG_ALPHA = 0.1

# ================================================================
# ================== NEW: ROI ALIGNMENT LINE =====================
# ================================================================
ROI_ALIGN_X = 250                 # x-position INSIDE the ROI (pixels)
ROI_ALIGN_COLOR = (0, 255, 255)   # BGR (yellow)
ROI_ALIGN_THICKNESS = 2
# ================================================================

# ================================================================
# ===================== NEW: COLOUR SPLITTING =====================
# ================================================================
CLASSIFY_BY = "y"          # "y" (upper/lower) or "x" (left/right)
CLASSIFY_FRAC = 0.5        # 0.5 = middle of ROI; 0.4 shifts line upwards, etc.

COLOR_A = "red"
COLOR_B = "blue"
PLOT_CLASSIFY_LINE = True
# ================================================================

# ================================================================
# ====================== FPS TEST SETTINGS ========================
# ================================================================
# Print FPS stats every this many seconds
FPS_PRINT_EVERY_S = 1.0

# Rolling FPS window length (seconds)
FPS_ROLLING_WINDOW_S = 2.0
# ================================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================================================================
# =================== CAMERA CONFIG FUNCTION =====================
# ================================================================
def configure_camera(cam):
    try:
        cam.set_pixel_format(PixelFormat.Mono8)
    except Exception as e:
        print("Could not set Mono8 pixel format:", e)

    for name in ("Width", "Height"):
        try:
            feat = cam.get_feature_by_name(name)
            rng = feat.get_range()
            max_v = rng[1]
            feat.set(max_v)
        except Exception as e:
            print(f"Could not set {name}:", e)

    try:
        w_feat = cam.get_feature_by_name("Width")
        h_feat = cam.get_feature_by_name("Height")
        offx_feat = cam.get_feature_by_name("OffsetX")
        offy_feat = cam.get_feature_by_name("OffsetY")

        w = w_feat.get()
        h = h_feat.get()

        max_offx = offx_feat.get_range()[1]
        max_offy = offy_feat.get_range()[1]

        offx_feat.set(max(0, min(max_offx, (max_offx - w) // 2)))
        offy_feat.set(max(0, min(max_offy, (max_offy - h) // 2)))
    except Exception:
        pass

    for name in ("BinningHorizontal", "BinningVertical",
                 "DecimationHorizontal", "DecimationVertical"):
        try:
            feat = cam.get_feature_by_name(name)
            feat.set(1)
        except Exception:
            pass

    try:
        exp = cam.get_feature_by_name("ExposureTime")
        min_e, max_e = exp.get_range()[0:2]
        exp.set(min(max_e * 0.3, max(min_e, 5000.0)))
    except Exception:
        pass

    try:
        gain = cam.get_feature_by_name("Gain")
        min_g, max_g = gain.get_range()[0:2]
        gain.set(min_g + (max_g - min_g) * 0.3)
    except Exception:
        pass

    print("Camera configured for high-quality capture.")

# ================================================================
# =================== PREPROCESSING FUNCTION =====================
# ================================================================
def preprocess_frame(frame):
    if frame.ndim == 3:
        if frame.shape[2] == 1:
            gray = frame[:, :, 0]
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

# ================================================================
# =============== DROPLET DETECTION FUNCTION =====================
# ================================================================
def detect_droplet(gray, last_center=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    src = cv2.bitwise_not(gray_eq) if DROPLET_IS_DARK else gray_eq

    thr = cv2.adaptiveThreshold(src, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,
                                21, -THRESH_VAL)

    kernel = np.ones((3, 3), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0

    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_AREA < area < MAX_AREA):
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w > MAX_BLOB_WIDTH or h > MAX_BLOB_HEIGHT:
            continue
        if w == 0 or h == 0:
            continue

        aspect = max(w, h) / min(w, h)
        if aspect > MAX_ASPECT_RATIO:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        if last_center is not None:
            lx, ly = last_center
            dist = np.hypot(cx - lx, cy - ly)
            if dist > MAX_JUMP_PIXELS:
                continue
            score = area / (1.0 + dist)
        else:
            score = area * circularity

        if score > best_score:
            best_score = score
            best = (cx, cy, area)

    return best, thr

# ================================================================
# ===================== FPS / DROPPED-FRAME HELPERS ===============
# ================================================================
def _get_frame_id_safe(frame):
    """
    Try several ways to extract a frame ID from VmbPy Frame.
    Returns int or None.
    """
    # Some versions expose frame_id attribute
    fid = getattr(frame, "frame_id", None)
    if fid is not None:
        try:
            return int(fid)
        except Exception:
            pass

    # Some versions use get_id()
    try:
        fid = frame.get_id()
        return int(fid)
    except Exception:
        pass

    # Some versions expose get_frame_id()
    try:
        fid = frame.get_frame_id()
        return int(fid)
    except Exception:
        pass

    return None

# ================================================================
# ===================== TRACK FROM CAMERA ========================
# ================================================================
def track_camera():
    trajectories = []
    traj_colors = []

    frame_idx = 0
    last_center = None
    miss_count = 0
    tracking_active = False
    current_traj = []

    current_color = None

    bg_roi = None
    bg_count = 0

    split_value_full = None

    # ---------------- FPS instrumentation ----------------
    t0 = time.perf_counter()
    last_print = t0
    n_frames = 0
    times = deque(maxlen=4000)  # timestamps for rolling FPS
    dropped_frames = 0
    last_fid = None
    fid_supported = None  # will become True/False after first few frames
    # -----------------------------------------------------

    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        if not cams:
            raise RuntimeError("No Allied Vision cameras detected.")

        cam = cams[0]

        with cam:
            configure_camera(cam)

            print("Starting streaming… Press Q in the video window to stop.")
            print(f"Building background from first {BG_INIT_FRAMES} frames "
                  "— do not create droplets yet.")

            for frame in cam.get_frame_generator():
                # ---------------- FPS instrumentation ----------------
                now = time.perf_counter()
                times.append(now)
                n_frames += 1

                fid = _get_frame_id_safe(frame)
                if fid_supported is None and n_frames >= 5:
                    # Decide support after a few frames
                    fid_supported = fid is not None
                    if fid_supported:
                        print("[Python] FrameID detected: dropped-frame check ENABLED")
                    else:
                        print("[Python] FrameID not available: dropped-frame check DISABLED")

                if fid is not None:
                    if last_fid is None:
                        last_fid = fid
                    else:
                        gap = fid - last_fid
                        if gap > 1:
                            dropped_frames += (gap - 1)
                        last_fid = fid

                # Print once per second
                if now - last_print >= FPS_PRINT_EVERY_S:
                    # overall average
                    avg_fps = n_frames / (now - t0)

                    # rolling over last FPS_ROLLING_WINDOW_S seconds
                    t_min = now - FPS_ROLLING_WINDOW_S
                    # fast rolling calc: find first index >= t_min
                    recent = [t for t in times if t >= t_min]
                    if len(recent) >= 2:
                        roll_fps = (len(recent) - 1) / (recent[-1] - recent[0])
                    else:
                        roll_fps = float("nan")

                    if fid_supported:
                        #print(f"[Python] avg FPS={avg_fps:.2f} | rolling({FPS_ROLLING_WINDOW_S:.0f}s) FPS={roll_fps:.2f} | dropped={dropped_frames}")
                   
                        #print(f"[Python] avg FPS={avg_fps:.2f} | rolling({FPS_ROLLING_WINDOW_S:.0f}s) FPS={roll_fps:.2f}")
                        last_print = now
                # -----------------------------------------------------

                img = frame.as_numpy_ndarray()
                h_img, w_img = img.shape[:2]

                if frame_idx == 0:
                    print("Frame shape from camera:", img.shape)

                x0_frac, y0_frac, x1_frac, y1_frac = ROI_FRAC
                x0 = int(x0_frac * w_img)
                x1 = int(x1_frac * w_img)
                y0 = int(y0_frac * h_img)
                y1 = int(y1_frac * h_img)

                if split_value_full is None:
                    if CLASSIFY_BY.lower() == "y":
                        split_value_full = y0 + int(CLASSIFY_FRAC * (y1 - y0))
                    else:
                        split_value_full = x0 + int(CLASSIFY_FRAC * (x1 - x0))

                roi = img[y0:y1, x0:x1]
                gray = preprocess_frame(roi)

                if bg_roi is None:
                    bg_roi = gray.astype(np.float32)
                    bg_count = 1
                    disp_roi = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                elif bg_count < BG_INIT_FRAMES:
                    cv2.accumulateWeighted(gray, bg_roi, BG_ALPHA)
                    bg_count += 1
                    disp_roi = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                else:
                    bg_u8 = cv2.convertScaleAbs(bg_roi)
                    diff = cv2.absdiff(gray, bg_u8)

                    droplet, thresh = detect_droplet(diff, last_center)

                    if droplet is not None:
                        cx, cy, area = droplet
                        last_center = (cx, cy)
                        miss_count = 0

                        full_x = x0 + cx
                        full_y = y0 + cy

                        if not tracking_active:
                            current_traj = []
                            tracking_active = True

                            if CLASSIFY_BY.lower() == "y":
                                current_color = COLOR_A if full_y < split_value_full else COLOR_B
                            else:
                                current_color = COLOR_A if full_x < split_value_full else COLOR_B

                        current_traj.append([frame_idx, full_x, full_y])

                    elif tracking_active and last_center is not None and miss_count < MAX_MISSES:
                        miss_count += 1
                        lx, ly = last_center
                        full_x = x0 + lx
                        full_y = y0 + ly
                        current_traj.append([frame_idx, full_x, full_y])

                    else:
                        if tracking_active:
                            if len(current_traj) >= MIN_TRACK_LENGTH:
                                trajectories.append(np.array(current_traj))
                                traj_colors.append(current_color if current_color is not None else "blue")
                                print(f"Stored droplet track with {len(current_traj)} points. Color={traj_colors[-1]}")
                            else:
                                print("Discarded very short track.")
                            tracking_active = False
                            current_traj = []
                            current_color = None

                        last_center = None
                        miss_count = 0

                    disp_roi = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
                    if last_center is not None:
                        lx, ly = last_center
                        cv2.circle(disp_roi, (int(lx), int(ly)), 8, (0, 0, 255), 2)

                if img.ndim == 2:
                    full_disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.ndim == 3 and img.shape[2] == 1:
                    full_disp = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
                else:
                    full_disp = img.copy()

                full_disp[y0:y1, x0:x1] = disp_roi
                cv2.rectangle(full_disp, (x0, y0), (x1, y1), (0, 255, 0), 1)

                if split_value_full is not None:
                    if CLASSIFY_BY.lower() == "y":
                        cv2.line(full_disp, (x0, split_value_full), (x1, split_value_full), (255, 255, 0), 1)
                    else:
                        cv2.line(full_disp, (split_value_full, y0), (split_value_full, y1), (255, 255, 0), 1)

                roi_w = x1 - x0
                if 0 <= ROI_ALIGN_X < roi_w:
                    x_align_full = x0 + ROI_ALIGN_X
                    cv2.line(full_disp,
                             (x_align_full, y0),
                             (x_align_full, y1),
                             ROI_ALIGN_COLOR,
                             ROI_ALIGN_THICKNESS)

                cv2.imshow("Live droplet tracking", full_disp)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_idx += 1

    cv2.destroyAllWindows()

    if tracking_active and len(current_traj) >= MIN_TRACK_LENGTH:
        trajectories.append(np.array(current_traj))
        traj_colors.append(current_color if current_color is not None else "blue")
        print(f"Stored last droplet track with {len(current_traj)} points. Color={traj_colors[-1]}")

    print(f"Total droplets tracked this run: {len(trajectories)}")
    return trajectories, traj_colors, split_value_full

# ================================================================
# ===================== GEOMETRY HELPERS =========================
# ================================================================
def y_at_x_first_crossing(x, y, x_target):
    x = np.asarray(x)
    y = np.asarray(y)

    for i in range(len(x) - 1):
        x0, x1 = x[i], x[i + 1]
        if (x0 - x_target) * (x1 - x_target) <= 0:
            if x1 == x0:
                return y[i]
            t = (x_target - x0) / (x1 - x0)
            return y[i] + t * (y[i + 1] - y[i])
    return None

def compute_impact_parameter(x, y, x_slit):
    return y_at_x_first_crossing(x, y, x_slit)

def wrap_to_90(angle_deg):
    return ((angle_deg + 90) % 180) - 90

def compute_deflection_angle_between_x(x, y, x_start, x_end):
    """
    Angle measured relative to the HORIZONTAL.

    - interpolate y(x_start) and y(x_end)
    - define dy = y_end - y_start, dx = x_end - x_start
    - theta_h = atan2(dy, dx)   (angle from horizontal)
    - wrap to [-90, +90]
    """
    y_start = y_at_x_first_crossing(x, y, x_start)
    y_end = y_at_x_first_crossing(x, y, x_end)

    if y_start is None or y_end is None:
        return None

    dx = x_end - x_start
    dy = y_end - y_start

    theta_h = np.degrees(np.arctan2(dy, dx))
    theta_h = wrap_to_90(theta_h)
    return theta_h

# ================================================================
# ============== NEW: SAVE LONG-FORMAT CSV PER RUN ================
# ================================================================
def save_long_format_csv(traj_list, run_id, fps_hz=None):
    """
    Save ONE CSV for the whole run:
      columns: run_id, droplet_id, frame, t_s, x_px, y_px
    """
    out_name = f"trajectories_long_{run_id}.csv"
    out_path = os.path.join(OUTPUT_FOLDER, out_name)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "droplet_id", "frame", "t_s", "x_px", "y_px"])

        for droplet_id, traj in enumerate(traj_list):
            # traj columns: [frame, x, y]
            for row in traj:
                frame = int(row[0])
                x_px = float(row[1])
                y_px = float(row[2])
                t_s = (frame / float(fps_hz)) if (fps_hz is not None and fps_hz > 0) else ""
                writer.writerow([run_id, droplet_id, frame, t_s, x_px, y_px])

    print("Saved long-format trajectories to:", out_path)
    return out_path

# ================================================================
# ======================== MAIN PIPELINE =========================
# ================================================================
def main():
    print("Starting live camera droplet tracking... Press Q in the video window to stop.")

    traj_list, traj_colors, split_value_full = track_camera()

    if not traj_list:
        print("No valid droplets tracked — tune detection / ROI / MIN_TRACK_LENGTH.")
        return

    # ------------------ NEW: save ONE long-format CSV per run ------------------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_long_format_csv(traj_list, run_id, fps_hz=FRAME_RATE_HZ)
    # -------------------------------------------------------------------------

    all_yi = []
    all_theta = []

    plt.figure(figsize=(5, 5))
    for i, (traj, col) in enumerate(zip(traj_list, traj_colors)):
        x = traj[:, 1]
        y = traj[:, 2]

        yi = compute_impact_parameter(x, y, X_SLIT_IN)
        theta_h = compute_deflection_angle_between_x(x, y, X_ANGLE_START, X_ANGLE_END)

        print(f"Droplet {i}: yi={yi}, theta(from horizontal)={theta_h}, color={col}")

        if yi is not None and theta_h is not None:
            all_yi.append(yi)
            all_theta.append(theta_h)

        plt.plot(x, y, lw=0.7, color=col, alpha=0.7)

    plt.axvline(X_SLIT_IN, color='blue', ls='--', label='x = X_SLIT_IN (impact)')
    plt.axvline(X_ANGLE_END, color='purple', ls='--', label='x = X_ANGLE_END (angle end)')

    if PLOT_CLASSIFY_LINE and split_value_full is not None:
        if CLASSIFY_BY.lower() == "y":
            plt.axhline(split_value_full, color='gold', ls='--', lw=1.0, label='colour split line')
        else:
            plt.axvline(split_value_full, color='gold', ls='--', lw=1.0, label='colour split line')

    plt.gca().invert_yaxis()
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.title("Droplet trajectories")
    plt.tight_layout()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    traj_fname = f"droplet_trajectories_{stamp}.png"
    traj_path = os.path.join(OUTPUT_FOLDER, traj_fname)
    #plt.savefig(traj_path, dpi=300)
    print("Saved trajectory plot to:", traj_path)
    plt.show()

    if not all_yi or not all_theta:
        print("Could not compute yi/theta for any droplet. Check X_SLIT_IN, X_ANGLE_START/END, and track lengths.")
        return

    all_yi = np.array(all_yi)
    all_theta = np.array(all_theta)

    plt.figure(figsize=(4, 4))
    plt.scatter(all_yi, all_theta, s=10, color='red')
    plt.xlabel("Impact parameter y_i (pixels)")
    plt.ylabel("Deflection angle θ (deg, from horizontal)")
    plt.grid(True)
    plt.title("θ vs y_i (all droplets)")
    plt.tight_layout()
    plt.show()

    all_theta = wrap_to_90(all_theta)
    bins = np.linspace(-90, 90, 61)

    plt.figure(figsize=(4, 4))
    plt.hist(all_theta, bins=bins, density=True)
    plt.xlim(-90, 90)
    plt.xlabel("θ (deg, from horizontal)")
    plt.xticks(np.arange(-90,90,20))
    plt.ylabel("P(θ)")
    plt.title("Deflection angle distribution")
    plt.tight_layout()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_fname = f"theta_hist_{stamp}.png"
    hist_path = os.path.join(OUTPUT_FOLDER, hist_fname)
    #plt.savefig(hist_path, dpi=300)
    print("Saved histogram plot to:", hist_path)
    plt.show()

    print(f"Analysed {len(all_theta)} droplets. Results saved in {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
