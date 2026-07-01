import cv2
import os
import time


# ========= USER SETTINGS =========
# CAMERA_SOURCE = 1 for Camo (iPhone via USB), or 0 for built-in webcam
CAMERA_SOURCE = 1

SAVE_DIR = "charuco_calib_images"
MIN_CORNERS_TO_SAVE = 12
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720

# Board definition - MUST match your printed board exactly
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH = 0.040      # meters
MARKER_LENGTH = 0.028      # meters

ARUCO_DICT = cv2.aruco.DICT_4X4_50
USE_LEGACY_PATTERN = False
# =================================


os.makedirs(SAVE_DIR, exist_ok=True)

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
if USE_LEGACY_PATTERN and hasattr(board, "setLegacyPattern"):
    board.setLegacyPattern(True)

detector_params = cv2.aruco.DetectorParameters()
charuco_detector = cv2.aruco.CharucoDetector(board)

# Use DirectShow backend on Windows for more stable virtual camera handling
cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError(f"Could not open camera source at index {CAMERA_SOURCE}.")

print("=== ChArUco Capture (Camo / USB) ===")
print("Keys:")
print("  s  -> save current frame if enough corners are detected")
print("  q  -> quit")
print(f"Saving into: {SAVE_DIR}")

save_count = 0
last_save_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Waiting for frame...")
        time.sleep(0.1)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
    except Exception:
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
        charuco_corners, charuco_ids = None, None
        if marker_ids is not None and len(marker_ids) > 0:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board
            )

    display = frame.copy()

    marker_count = 0 if marker_ids is None else len(marker_ids)
    charuco_count = 0 if charuco_ids is None else len(charuco_ids)

    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)

    if charuco_ids is not None and len(charuco_ids) > 0:
        cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids, (0, 255, 0))

    ok_to_save = charuco_count >= MIN_CORNERS_TO_SAVE

    cv2.putText(display, f"Markers: {marker_count}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(display, f"ChArUco corners: {charuco_count}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(display, f"Save status: {'READY' if ok_to_save else 'NOT READY'}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if ok_to_save else (0, 0, 255), 2)
    cv2.putText(display, "Press 's' to save, 'q' to quit", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("ChArUco Capture", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('s'):
        now = time.time()
        if not ok_to_save:
            print(f"Skipped: only {charuco_count} corners detected.")
            continue
        if now - last_save_time < 0.5:
            continue

        filename = os.path.join(SAVE_DIR, f"img_{save_count:03d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}  | corners={charuco_count}")
        save_count += 1
        last_save_time = now

cap.release()
cv2.destroyAllWindows()
print(f"Done. Saved {save_count} images.")
