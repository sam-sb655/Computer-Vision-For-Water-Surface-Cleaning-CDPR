import cv2
import numpy as np
import os
import glob

# ========= USER SETTINGS =========
IMAGE_DIR = "charuco_calib_images"
OUTPUT_FILE = "phonecameracalib.npz"

# Board definition - MUST match your printed board exactly
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH = 0.040      # meters
MARKER_LENGTH = 0.028      # meters
ARUCO_DICT = cv2.aruco.DICT_4X4_50

USE_LEGACY_PATTERN = False

MIN_CORNERS_PER_IMAGE = 10
SHOW_DEBUG = True
# =================================

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
if USE_LEGACY_PATTERN and hasattr(board, "setLegacyPattern"):
    board.setLegacyPattern(True)

detector_params = cv2.aruco.DetectorParameters()
charuco_detector = cv2.aruco.CharucoDetector(board)

image_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
image_paths = sorted(image_paths)

if len(image_paths) == 0:
    raise RuntimeError(f"No calibration images found in {IMAGE_DIR}")

all_charuco_corners = []
all_charuco_ids = []
used_images = []
image_size = None

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"Skipping unreadable image: {path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = (gray.shape[1], gray.shape[0])

    try:
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
    except Exception:
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
        charuco_corners, charuco_ids = None, None
        if marker_ids is not None and len(marker_ids) > 0:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board
            )

    n = 0 if charuco_ids is None else len(charuco_ids)

    if charuco_ids is not None and n >= MIN_CORNERS_PER_IMAGE:
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        used_images.append(path)

        if SHOW_DEBUG:
            dbg = img.copy()
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(dbg, marker_corners, marker_ids)
            cv2.aruco.drawDetectedCornersCharuco(dbg, charuco_corners, charuco_ids, (0, 255, 0))
            cv2.imshow("Accepted calibration image", dbg)
            cv2.waitKey(80)
    else:
        print(f"Rejected {os.path.basename(path)}: only {n} ChArUco corners")

cv2.destroyAllWindows()

if len(all_charuco_corners) < 8:
    raise RuntimeError(f"Too few valid images for calibration: {len(all_charuco_corners)}")

flags = 0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 200, 1e-9)

rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_charuco_corners,
    charucoIds=all_charuco_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None,
    flags=flags,
    criteria=criteria
)

# Per-image reprojection errors
per_view_errors = []
for corners, ids, rvec, tvec in zip(all_charuco_corners, all_charuco_ids, rvecs, tvecs):
    obj_points, img_points = board.matchImagePoints(corners, ids)
    projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    img_points = img_points.reshape(-1, 2)
    err = np.mean(np.linalg.norm(projected - img_points, axis=1))
    per_view_errors.append(err)

per_view_errors = np.array(per_view_errors, dtype=np.float64)

np.savez(
    OUTPUT_FILE,
    cameraMatrix=camera_matrix,
    distCoeffs=dist_coeffs,
    imageSize=np.array(image_size),
    rms=float(rms),
    perViewErrors=per_view_errors,
    usedImages=np.array(used_images, dtype=object),
    squaresX=SQUARES_X,
    squaresY=SQUARES_Y,
    squareLength=float(SQUARE_LENGTH),
    markerLength=float(MARKER_LENGTH),
    arucoDict=int(ARUCO_DICT),
    useLegacyPattern=bool(USE_LEGACY_PATTERN),
)

print("\n=== Calibration complete ===")
print(f"Valid images used: {len(used_images)} / {len(image_paths)}")
print(f"Image size: {image_size}")
print(f"RMS reprojection error: {rms:.4f} px")
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coeffs:\n", dist_coeffs.ravel())
print(f"Mean per-view error: {per_view_errors.mean():.4f} px")
print(f"Max per-view error : {per_view_errors.max():.4f} px")
print(f"Saved to: {OUTPUT_FILE}")
