import cv2
import numpy as np
import socket
import time
import json
import csv
from collections import deque

# =========================================================
# CAMERA / NETWORK
# =========================================================
CAM_INDEX = 1
CALIB_FILE = "phonecameracalib.npz"
MATLAB_IP = "10.50.52.169"
MATLAB_PORT = 5005

# =========================================================
# MANUAL TARGET POINT
# Target is specified in REPORTED aquarium coordinates
# i.e. coordinates that include ORIGIN offsets, not EE offsets
# =========================================================
TARGET_X_WORLD = 20.0
TARGET_Y_WORLD = 15.0

# =========================================================
# CHARUCO DIAMOND GEOMETRY
# =========================================================
SQUARE_LENGTH_CM = 4.1
MARKER_LENGTH_CM = 2.4
SQUARE_MARKER_RATIO = SQUARE_LENGTH_CM / MARKER_LENGTH_CM

# =========================================================
# FILTER / VALIDATION
# =========================================================
MAX_REPROJ_ERR_PX = 3.0
MIN_MARKER_AREA_PX = 700.0
EMA_ALPHA = 0.18

EE_OFFSET_X = 1.7
EE_OFFSET_Y = 0.5
EE_OFFSET_Z = 9.0

ORIGIN_OFFSET_X = 5.5
ORIGIN_OFFSET_Y = 6.6
ORIGIN_OFFSET_Z = 3.2

MAX_PREDICT_HOLD_S = 0.20
MAX_REASONABLE_JUMP_CM = 2.0

# =========================================================
# CALIBRATION SETTINGS
# Hold the initial corner pose until GO appears
# =========================================================
CALIB_MIN_VALID_FRAMES = 45
CALIB_POS_STD_THRESH_CM = 0.20
CALIB_AXIS_STD_THRESH_DEG = 1.20

# =========================================================
# IDS
# =========================================================
ID_MAP = {
    (0, 1, 2, 3): "P1",
    (4, 5, 6, 7): "P2",
    (8, 9, 10, 11): "P3",
    (12, 13, 14, 15): "P4",
    (16, 17, 18, 19): "EE",
}

WORKSPACE_ANCHORS = ["P1", "P2", "P3", "P4"]
TRACK_REQUIRED = ["EE"]

half_sq = SQUARE_LENGTH_CM / 2.0
OBJ_PTS = np.array([
    [-half_sq,  half_sq, 0.0],
    [ half_sq,  half_sq, 0.0],
    [ half_sq, -half_sq, 0.0],
    [-half_sq, -half_sq, 0.0]
], dtype=np.float32)

# =========================================================
# LOAD CALIBRATION
# =========================================================
with np.load(CALIB_FILE) as d:
    print("Calibration file keys:", d.files)

    if "camera_matrix" in d:
        K = d["camera_matrix"]
    elif "cameraMatrix" in d:
        K = d["cameraMatrix"]
    else:
        raise KeyError("No camera matrix key found in calibration file")

    if "dist_coeffs" in d:
        DC = d["dist_coeffs"]
    elif "distCoeffs" in d:
        DC = d["distCoeffs"]
    else:
        raise KeyError("No distortion coeffs key found in calibration file")

# =========================================================
# ARUCO / CHARUCO DETECTOR
# =========================================================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()

params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.cornerRefinementWinSize = 5
params.cornerRefinementMaxIterations = 40
params.cornerRefinementMinAccuracy = 0.01
params.minDistanceToBorder = 8

params.polygonalApproxAccuracyRate = 0.04
params.adaptiveThreshWinSizeMin = 5
params.adaptiveThreshWinSizeMax = 21
params.adaptiveThreshWinSizeStep = 8
params.adaptiveThreshConstant = 7
params.minMarkerPerimeterRate = 0.04
params.maxMarkerPerimeterRate = 4.0

detector = cv2.aruco.ArucoDetector(aruco_dict, params)

# =========================================================
# UDP
# =========================================================
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# =========================================================
# KALMAN FILTER
# state = [x y z vx vy vz] in REPORTED aquarium coordinates
# =========================================================
kf = cv2.KalmanFilter(6, 3)
kf.transitionMatrix = np.eye(6, dtype=np.float32)
kf.measurementMatrix = np.zeros((3, 6), dtype=np.float32)
kf.measurementMatrix[0, 0] = 1.0
kf.measurementMatrix[1, 1] = 1.0
kf.measurementMatrix[2, 2] = 1.0
kf.processNoiseCov = np.diag([1.0e-3, 1.0e-3, 1.0e-3, 1.5e-2, 1.5e-2, 1.5e-2]).astype(np.float32)
kf.measurementNoiseCov = np.diag([2.0e-2, 2.0e-2, 4.0e-2]).astype(np.float32)
kf.errorCovPost = np.eye(6, dtype=np.float32) * 10.0

kf_initialized = False
ema_state = None
prev_time = time.time()
t0 = prev_time
last_good_meas_time = None
last_sent_xyz = None

traj_t = deque(maxlen=41)
traj_x = deque(maxlen=41)
traj_y = deque(maxlen=41)
traj_z = deque(maxlen=41)

csv_file = open("cv_x_validation_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["t", "x", "y", "z", "vx", "ax", "valid"])

# =========================================================
# CALIBRATION STATE
# =========================================================
calib_samples = []
calibration_locked = False
T_aq_to_cam = None
T_cam_to_aq = None
R_aq_to_cam = None
t_aq_to_cam = None

target_locked_px = None
target_aq_xy = np.array([
    TARGET_X_WORLD - ORIGIN_OFFSET_X,
    TARGET_Y_WORLD - ORIGIN_OFFSET_Y
], dtype=np.float32)

# =========================================================
# HELPERS
# =========================================================
def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return None
    return v / n

def polygon_area(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def solve_best_pose(img_pts):
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        OBJ_PTS,
        img_pts.reshape(4, 1, 2).astype(np.float32),
        K,
        DC,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok or len(rvecs) == 0:
        return None

    best = None
    best_err = 1e9
    for i in range(len(rvecs)):
        rvec = rvecs[i]
        tvec = tvecs[i]
        z = float(tvec[2][0])
        if z <= 0:
            continue

        proj, _ = cv2.projectPoints(OBJ_PTS, rvec, tvec, K, DC)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)))

        if err < best_err:
            best_err = err
            best = (rvec, tvec.reshape(3), err)

    return best

def moving_average(arr, win=9):
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) < win:
        return arr.copy()
    kernel = np.ones(win, dtype=np.float64) / win
    pad = win // 2
    padded = np.pad(arr, (pad, pad), mode='edge')
    return np.convolve(padded, kernel, mode='valid')

def central_diff_uniform_like(t, x):
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(t)

    if n < 5:
        return None, None

    xs = moving_average(x, win=9)
    vx = np.full(n, np.nan)
    ax = np.full(n, np.nan)

    for i in range(1, n - 1):
        dt2 = t[i + 1] - t[i - 1]
        if dt2 > 1e-6:
            vx[i] = (xs[i + 1] - xs[i - 1]) / dt2

    finite_idx = np.where(np.isfinite(vx))[0]
    if len(finite_idx) >= 5:
        vxs = moving_average(vx[finite_idx], win=5)
        vx2 = vx.copy()
        if len(vxs) == len(finite_idx):
            vx2[finite_idx] = vxs
        else:
            vx2 = vx
    else:
        vx2 = vx

    for i in range(2, n - 2):
        dt2 = t[i + 1] - t[i - 1]
        if np.isfinite(vx2[i - 1]) and np.isfinite(vx2[i + 1]) and dt2 > 1e-6:
            ax[i] = (vx2[i + 1] - vx2[i - 1]) / dt2

    finite_ax = np.isfinite(ax)
    if np.sum(finite_ax) >= 5:
        ax_s = moving_average(ax[finite_ax], win=5)
        ax_out = ax.copy()
        if len(ax_s) == np.sum(finite_ax):
            ax_out[finite_ax] = ax_s
        else:
            ax_out = ax
    else:
        ax_out = ax

    return vx2, ax_out

def build_frame_from_p1_p2_p4(p1, p2, p4):
    ex = normalize(p4 - p1)
    if ex is None:
        return None

    y_raw = p2 - p1
    ez = normalize(np.cross(ex, y_raw))
    if ez is None:
        return None

    ey = normalize(np.cross(ez, ex))
    if ey is None:
        return None

    R = np.column_stack((ex, ey, ez)).astype(np.float32)
    return R

def make_homogeneous(R, t):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

def invert_homogeneous(T):
    R = T[:3, :3]
    t = T[:3, 3].reshape(3, 1)
    Tinv = np.eye(4, dtype=np.float32)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = (-R.T @ t).reshape(3)
    return Tinv

def transform_point(T, p3):
    ph = np.array([p3[0], p3[1], p3[2], 1.0], dtype=np.float32)
    out = T @ ph
    return out[:3]

def angle_deg_between(u, v):
    u = normalize(u)
    v = normalize(v)
    if u is None or v is None:
        return 180.0
    c = np.clip(np.dot(u, v), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def try_lock_calibration(samples):
    if len(samples) < CALIB_MIN_VALID_FRAMES:
        return False, None, None, None

    p1s = np.array([s["p1"] for s in samples], dtype=np.float32)
    exs = np.array([s["ex"] for s in samples], dtype=np.float32)
    eys = np.array([s["ey"] for s in samples], dtype=np.float32)

    p1_mean = np.mean(p1s, axis=0)
    ex_mean_raw = np.mean(exs, axis=0)
    ey_mean_raw = np.mean(eys, axis=0)

    ex_mean = normalize(ex_mean_raw)
    if ex_mean is None:
        return False, None, None, None

    ez_mean = normalize(np.cross(ex_mean, ey_mean_raw))
    if ez_mean is None:
        return False, None, None, None

    ey_mean = normalize(np.cross(ez_mean, ex_mean))
    if ey_mean is None:
        return False, None, None, None

    pos_std = np.mean(np.std(p1s, axis=0))
    ex_ang = np.mean([angle_deg_between(v, ex_mean) for v in exs])
    ey_ang = np.mean([angle_deg_between(v, ey_mean) for v in eys])

    if pos_std > CALIB_POS_STD_THRESH_CM:
        return False, pos_std, ex_ang, ey_ang
    if ex_ang > CALIB_AXIS_STD_THRESH_DEG or ey_ang > CALIB_AXIS_STD_THRESH_DEG:
        return False, pos_std, ex_ang, ey_ang

    R = np.column_stack((ex_mean, ey_mean, ez_mean)).astype(np.float32)
    t = p1_mean.astype(np.float32)

    Taq2cam = make_homogeneous(R, t)
    Tcam2aq = invert_homogeneous(Taq2cam)

    return True, (Taq2cam, Tcam2aq, R, t), pos_std, max(ex_ang, ey_ang)

def order_points_cyclic(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]

def draw_workspace_polygon(img, anchor_px_dict):
    if not all(name in anchor_px_dict for name in WORKSPACE_ANCHORS):
        return False, None

    pts = np.array([anchor_px_dict[name] for name in WORKSPACE_ANCHORS], dtype=np.float32)
    ordered = order_points_cyclic(pts)
    poly = np.round(ordered).astype(np.int32).reshape((-1, 1, 2))

    overlay = img.copy()
    cv2.fillPoly(overlay, [poly], (255, 0, 255))
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
    cv2.polylines(img, [poly], True, (255, 0, 255), 2, cv2.LINE_AA)

    return True, ordered

def line_value(A, B, P):
    x1, y1 = A
    x2, y2 = B
    xp, yp = P
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return a * xp + b * yp + c

def point_in_convex_polygon_same_side(P, poly, eps=1e-6):
    vals = []
    n = len(poly)
    for i in range(n):
        A = poly[i]
        B = poly[(i + 1) % n]
        vals.append(float(line_value(A, B, P)))
    vals = np.array(vals, dtype=np.float64)

    non_pos = np.all(vals <= eps)
    non_neg = np.all(vals >= -eps)
    inside = bool(non_pos or non_neg)
    return inside, vals

def camera_to_aquarium(p_cam):
    return transform_point(T_cam_to_aq, p_cam)

def aquarium_to_camera(p_aq):
    return transform_point(T_aq_to_cam, p_aq)

def ee_marker_aq_to_report(aq_xyz):
    x = float(aq_xyz[0]) - EE_OFFSET_X + ORIGIN_OFFSET_X
    y = float(aq_xyz[1]) - EE_OFFSET_Y + ORIGIN_OFFSET_Y
    z = 47.0 + (float(aq_xyz[2]) + (ORIGIN_OFFSET_Z - EE_OFFSET_Z))
    return np.array([x, y, z], dtype=np.float32)

def project_camera_point_to_image(cam_pt):
    if cam_pt[2] <= 1e-6:
        return None
    img_pt, _ = cv2.projectPoints(
        cam_pt.reshape(1, 1, 3).astype(np.float32),
        np.zeros((3, 1), dtype=np.float32),
        np.zeros((3, 1), dtype=np.float32),
        K,
        DC
    )
    return img_pt.reshape(2)

# =========================================================
# START
# =========================================================
print("\n--- WIRED CV TRACKER STARTED ---")

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

WINDOW_NAME = "Wired CV Tracker"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\r[WARN] Waiting for camera...", end="", flush=True)
            time.sleep(0.05)
            continue

        now = time.time()
        dt = max(now - prev_time, 1e-3)
        prev_time = now
        t_rel = now - t0

        kf.transitionMatrix[0, 3] = dt
        kf.transitionMatrix[1, 4] = dt
        kf.transitionMatrix[2, 5] = dt

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners_list, ids, _ = detector.detectMarkers(gray)

        poses = {}
        found = []
        anchor_px = {}

        if ids is not None and len(ids) >= 4:
            cv2.aruco.drawDetectedMarkers(display, corners_list, ids)

            try:
                dia_corners, dia_ids = cv2.aruco.detectCharucoDiamond(
                    gray, corners_list, ids, SQUARE_MARKER_RATIO
                )
            except Exception:
                dia_corners, dia_ids = [], None

            if dia_ids is not None and len(dia_ids) > 0:
                for i in range(len(dia_ids)):
                    id_tuple = tuple(int(x) for x in np.array(dia_ids[i]).flatten())
                    asset = ID_MAP.get(id_tuple, None)
                    if asset is None:
                        continue

                    found.append(asset)

                    img_pts = np.array(dia_corners[i], dtype=np.float32).reshape(4, 2)
                    area = polygon_area(img_pts)
                    if area < MIN_MARKER_AREA_PX:
                        continue

                    solved = solve_best_pose(img_pts)
                    if solved is None:
                        continue

                    rvec, center_cam, err = solved
                    if err > MAX_REPROJ_ERR_PX:
                        continue

                    poses[asset] = center_cam

                    c = np.mean(img_pts, axis=0).astype(np.float32)
                    if asset in WORKSPACE_ANCHORS:
                        anchor_px[asset] = c

                    c_int = tuple(np.round(c).astype(int))
                    cv2.circle(display, c_int, 5, (0, 255, 0), -1)
                    cv2.putText(
                        display,
                        f"{asset} e={err:.2f}",
                        (c_int[0] + 8, c_int[1] + 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )

        # =================================================
        # CALIBRATION PHASE
        # =================================================
        if not calibration_locked:
            if all(k in poses for k in ["P1", "P2", "P4"]):
                p1 = poses["P1"]
                p2 = poses["P2"]
                p4 = poses["P4"]

                R_try = build_frame_from_p1_p2_p4(p1, p2, p4)
                if R_try is not None:
                    calib_samples.append({
                        "p1": p1.copy(),
                        "ex": R_try[:, 0].copy(),
                        "ey": R_try[:, 1].copy()
                    })

                    if len(calib_samples) > CALIB_MIN_VALID_FRAMES + 20:
                        calib_samples = calib_samples[-(CALIB_MIN_VALID_FRAMES + 20):]

                    ok, payload, pos_std, ang_std = try_lock_calibration(calib_samples)

                    if ok:
                        T_aq_to_cam, T_cam_to_aq, R_aq_to_cam, t_aq_to_cam = payload
                        calibration_locked = True

                        target_aq_3d = np.array([target_aq_xy[0], target_aq_xy[1], 0.0], dtype=np.float32)
                        target_cam_3d = aquarium_to_camera(target_aq_3d)
                        target_px = project_camera_point_to_image(target_cam_3d)
                        if target_px is not None:
                            target_locked_px = target_px.copy()

            progress = min(len(calib_samples), CALIB_MIN_VALID_FRAMES)
            cv2.putText(
                display,
                "CALIBRATION MODE: keep anchors at tank corners",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                display,
                f"Stable frames: {progress}/{CALIB_MIN_VALID_FRAMES}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            if len(calib_samples) >= 5:
                p1s = np.array([s["p1"] for s in calib_samples], dtype=np.float32)
                exs = np.array([s["ex"] for s in calib_samples], dtype=np.float32)
                eys = np.array([s["ey"] for s in calib_samples], dtype=np.float32)

                p1_mean = np.mean(p1s, axis=0)
                ex_mean = normalize(np.mean(exs, axis=0))
                ey_mean = normalize(np.mean(eys, axis=0))

                pos_std_now = float(np.mean(np.std(p1s, axis=0)))
                ex_ang_now = float(np.mean([angle_deg_between(v, ex_mean) for v in exs])) if ex_mean is not None else 999.0
                ey_ang_now = float(np.mean([angle_deg_between(v, ey_mean) for v in eys])) if ey_mean is not None else 999.0
                ang_now = max(ex_ang_now, ey_ang_now)

                cv2.putText(
                    display,
                    f"P1 std: {pos_std_now:.3f} cm   Axis std: {ang_now:.3f} deg",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.64,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
            else:
                cv2.putText(
                    display,
                    "Waiting for valid P1, P2, P4 detections...",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.64,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA
                )

        # =================================================
        # RECONFIGURATION / TRACKING PHASE
        # =================================================
        else:
            cv2.putText(
                display,
                "GO SIGNAL: aquarium frame locked, ready to reconfigure anchors",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.74,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            workspace_ok, workspace_px_pts = draw_workspace_polygon(display, anchor_px)

            # Current workspace in fixed aquarium frame
            workspace_aq_xy = None
            if all(k in poses for k in WORKSPACE_ANCHORS):
                aq_pts = []
                for name in WORKSPACE_ANCHORS:
                    paq = camera_to_aquarium(poses[name])
                    aq_pts.append([paq[0], paq[1]])
                workspace_aq_xy = order_points_cyclic(np.array(aq_pts, dtype=np.float32))

            # Fixed target test
            if target_locked_px is not None:
                tp = tuple(np.round(target_locked_px).astype(int))

                if workspace_aq_xy is not None:
                    target_inside, vals = point_in_convex_polygon_same_side(target_aq_xy, workspace_aq_xy)
                    color = (0, 255, 0) if target_inside else (0, 0, 255)
                    status_text = "REACHABLE" if target_inside else "NO MORE REACHABLE"

                    cv2.circle(display, tp, 7, color, -1)
                    cv2.circle(display, tp, 12, color, 2)
                    cv2.putText(
                        display,
                        "TARGET",
                        (tp[0] + 10, tp[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA
                    )

                    cv2.putText(
                        display,
                        f"Target status: {status_text}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        color,
                        2,
                        cv2.LINE_AA
                    )

                    cv2.putText(
                        display,
                        f"Target aquarium XY: ({target_aq_xy[0]:.2f}, {target_aq_xy[1]:.2f}) cm",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.64,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    cv2.putText(
                        display,
                        f"L1:{vals[0]:.2f}  L2:{vals[1]:.2f}  L3:{vals[2]:.2f}  L4:{vals[3]:.2f}",
                        (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (200, 255, 200),
                        2,
                        cv2.LINE_AA
                    )
                else:
                    cv2.circle(display, tp, 7, (255, 255, 255), -1)
                    cv2.circle(display, tp, 12, (255, 255, 255), 2)
                    cv2.putText(
                        display,
                        "TARGET",
                        (tp[0] + 10, tp[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )
                    cv2.putText(
                        display,
                        "Target status: workspace incomplete",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA
                    )

            # -------------------------------------------------
            # EE tracking in fixed aquarium frame
            # -------------------------------------------------
            predicted = kf.predict()
            got_measurement = False
            status = ""

            if "EE" in poses:
                ee_cam = poses["EE"]
                ee_aq = camera_to_aquarium(ee_cam)
                meas_corr = ee_marker_aq_to_report(ee_aq)

                if last_sent_xyz is None:
                    accept_meas = True
                else:
                    jump = np.linalg.norm(meas_corr - last_sent_xyz)
                    accept_meas = jump < MAX_REASONABLE_JUMP_CM

                if accept_meas or not kf_initialized:
                    meas = np.array([
                        [meas_corr[0]],
                        [meas_corr[1]],
                        [meas_corr[2]]
                    ], dtype=np.float32)
                    got_measurement = True
                    last_good_meas_time = now
                else:
                    status = f"Measurement rejected: jump {jump:.2f} cm"
            else:
                status = "Missing: EE"

            if got_measurement:
                if not kf_initialized:
                    kf.statePost = np.array([
                        [meas[0, 0]],
                        [meas[1, 0]],
                        [meas[2, 0]],
                        [0.0],
                        [0.0],
                        [0.0]
                    ], dtype=np.float32)
                    kf_initialized = True
                    ema_state = np.array([
                        float(meas[0, 0]),
                        float(meas[1, 0]),
                        float(meas[2, 0])
                    ], dtype=np.float32)
                else:
                    corrected = kf.correct(meas)
                    kf_xyz = np.array([
                        float(corrected[0, 0]),
                        float(corrected[1, 0]),
                        float(corrected[2, 0])
                    ], dtype=np.float32)
                    ema_state = EMA_ALPHA * kf_xyz + (1.0 - EMA_ALPHA) * ema_state

                use_state = ema_state.copy()
                track_valid = True
                track_mode = "MEAS"

            else:
                if kf_initialized and last_good_meas_time is not None and (now - last_good_meas_time) <= MAX_PREDICT_HOLD_S:
                    pred_xyz = np.array([
                        float(predicted[0, 0]),
                        float(predicted[1, 0]),
                        float(predicted[2, 0])
                    ], dtype=np.float32)

                    if ema_state is None:
                        ema_state = pred_xyz.copy()
                    else:
                        ema_state = EMA_ALPHA * pred_xyz + (1.0 - EMA_ALPHA) * ema_state

                    use_state = ema_state.copy()
                    track_valid = True
                    track_mode = "PRED"
                else:
                    track_valid = False
                    track_mode = "LOST"

            if track_valid:
                corr_x, corr_y, corr_z = map(float, use_state)
                last_sent_xyz = use_state.copy()

                traj_t.append(t_rel)
                traj_x.append(corr_x)
                traj_y.append(corr_y)
                traj_z.append(corr_z)

                vx = 0.0
                ax = 0.0

                if len(traj_t) >= 9:
                    vx_all, ax_all = central_diff_uniform_like(list(traj_t), list(traj_x))
                    if vx_all is not None and np.isfinite(vx_all[-2]):
                        vx = float(vx_all[-2])
                    if ax_all is not None and np.isfinite(ax_all[-3]):
                        ax = float(ax_all[-3])

                packet = {
                    "t": round(t_rel, 4),
                    "x": round(corr_x, 3),
                    "y": round(corr_y, 3),
                    "z": round(corr_z, 3),
                    "vx": round(vx, 4),
                    "ax": round(ax, 4),
                    "valid": True
                }

                try:
                    udp_sock.sendto((json.dumps(packet) + "\n").encode("utf-8"), (MATLAB_IP, MATLAB_PORT))
                except Exception as e:
                    print(f"\n[UDP ERROR] {e}")

                csv_writer.writerow([
                    round(t_rel, 4),
                    round(corr_x, 4),
                    round(corr_y, 4),
                    round(corr_z, 4),
                    round(vx, 4),
                    round(ax, 4),
                    1
                ])

                msg1 = f"{track_mode} | X:{corr_x:6.2f} Y:{corr_y:6.2f} Z:{corr_z:6.2f}"
                msg2 = f"VX:{vx:7.3f} cm/s AX:{ax:7.3f} cm/s^2"
                cv2.putText(display, msg1, (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 255), 2)
                cv2.putText(display, msg2, (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 0), 2)
                print(f"\r{msg1} | {msg2}", end="", flush=True)

            else:
                try:
                    udp_sock.sendto(
                        (json.dumps({"t": round(t_rel, 4), "valid": False}) + "\n").encode("utf-8"),
                        (MATLAB_IP, MATLAB_PORT)
                    )
                except Exception:
                    pass

                csv_writer.writerow([round(t_rel, 4), "", "", "", "", "", 0])

                line1 = "Found: " + ",".join(found) if found else "Found: none"
                line2 = status if status else "No valid pose"
                cv2.putText(display, line1, (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
                cv2.putText(display, line2, (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
                print(f"\r{line1} | {line2}", end="", flush=True)

        cv2.imshow(WINDOW_NAME, display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    udp_sock.close()
    print("\n[INFO] Tracker terminated.")
