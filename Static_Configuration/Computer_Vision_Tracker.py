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
MATLAB_IP = "192.168.137.179"
MATLAB_PORT = 5005

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

# Wired setup: much tighter than Wi-Fi
MAX_PREDICT_HOLD_S = 0.20
MAX_REASONABLE_JUMP_CM = 8.0

ID_MAP = {
    (0, 1, 2, 3): "P1",
    (4, 5, 6, 7): "P2",
    (8, 9, 10, 11): "P3",
    (12, 13, 14, 15): "P4",
    (16, 17, 18, 19): "EE",
}

REQUIRED = ["P1", "P2", "P4", "EE"]

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

# Wired setup tuning: keep subpixel, tighten a bit versus Wi-Fi
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
# state = [x y z vx vy vz]
# measurement = [x y z]
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

    return vx2, ax_out

def local_to_world(local_xyz):
    x = float(local_xyz[0]) - EE_OFFSET_X + ORIGIN_OFFSET_X
    y = float(local_xyz[1]) - EE_OFFSET_Y + ORIGIN_OFFSET_Y
    z = 47.0 + (float(local_xyz[2]) + (ORIGIN_OFFSET_Z - EE_OFFSET_Z))
    return np.array([x, y, z], dtype=np.float32)

# =========================================================
# START
# =========================================================
print("\n--- WIRED CV TRACKER STARTED ---")

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

WINDOW_NAME = "Wired CV Tracker"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1100, 600)

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

                    c = np.mean(img_pts, axis=0).astype(int)
                    cv2.circle(display, tuple(c), 5, (0, 255, 0), -1)
                    cv2.putText(
                        display,
                        f"{asset} e={err:.2f}",
                        (c[0] + 8, c[1] + 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1
                    )

        predicted = kf.predict()

        got_measurement = False
        status = ""

        if all(k in poses for k in REQUIRED):
            p1 = poses["P1"]
            p2 = poses["P2"]
            p4 = poses["P4"]
            ee = poses["EE"]

            ex = normalize(p4 - p1)
            y_raw = p2 - p1
            ez = normalize(np.cross(ex, y_raw)) if ex is not None else None
            ey = normalize(np.cross(ez, ex)) if (ez is not None and ex is not None) else None

            if ex is not None and ey is not None and ez is not None:
                R_local_to_cam = np.column_stack((ex, ey, ez))
                ee_rel_local = R_local_to_cam.T @ (ee - p1)

                meas_xyz = np.array([
                    float(ee_rel_local[0]),
                    float(ee_rel_local[1]),
                    float(ee_rel_local[2])
                ], dtype=np.float32)

                meas_corr = local_to_world(meas_xyz)

                if last_sent_xyz is None:
                    accept_meas = True
                else:
                    jump = np.linalg.norm(meas_corr - last_sent_xyz)
                    accept_meas = jump < MAX_REASONABLE_JUMP_CM

                if accept_meas or not kf_initialized:
                    meas = np.array([
                        [meas_xyz[0]],
                        [meas_xyz[1]],
                        [meas_xyz[2]]
                    ], dtype=np.float32)
                    got_measurement = True
                    last_good_meas_time = now
                else:
                    status = f"Measurement rejected: jump {jump:.2f} cm"
            else:
                status = "Frame fail: local frame degenerate"
        else:
            missing = [k for k in REQUIRED if k not in poses]
            status = "Missing: " + ",".join(missing)

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
            current_xyz = local_to_world(use_state)
            corr_x, corr_y, corr_z = map(float, current_xyz)
            last_sent_xyz = current_xyz.copy()

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

            msg1 = f"{track_mode} | X:{corr_x:6.2f}  Y:{corr_y:6.2f}  Z:{corr_z:6.2f}"
            msg2 = f"VX:{vx:7.3f} cm/s   AX:{ax:7.3f} cm/s^2"
            cv2.putText(display, msg1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, msg2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
            print(f"\r{msg1} | {msg2}", end="", flush=True)

        else:
            try:
                udp_sock.sendto((json.dumps({"t": round(t_rel, 4), "valid": False}) + "\n").encode("utf-8"),
                                (MATLAB_IP, MATLAB_PORT))
            except Exception:
                pass

            csv_writer.writerow([round(t_rel, 4), "", "", "", "", "", 0])

            line1 = "Found: " + ",".join(found) if found else "Found: none"
            line2 = status if status else "No valid pose"
            cv2.putText(display, line1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, line2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
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
