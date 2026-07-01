import cv2
import numpy as np

CAM_INDEX = 1

SQUARELENGTHCM = 4.1
MARKERLENGTHCM = 2.4
SQUAREMARKERRATIO = SQUARELENGTHCM / MARKERLENGTHCM

TARGET_IDS = (16, 17, 18, 19)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.cornerRefinementWinSize = 7
params.cornerRefinementMaxIterations = 40
params.cornerRefinementMinAccuracy = 0.01
params.minDistanceToBorder = 5
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 23
params.adaptiveThreshWinSizeStep = 10
params.adaptiveThreshConstant = 7
params.minMarkerPerimeterRate = 0.015
params.maxMarkerPerimeterRate = 4.0
params.polygonalApproxAccuracyRate = 0.05

detector = cv2.aruco.ArucoDetector(aruco_dict, params)

cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

cv2.namedWindow("Single ChArUco Diamond Detect", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners_list, ids, _ = detector.detectMarkers(gray)

    found_target = False

    if ids is not None and len(ids) >= 4:
        cv2.aruco.drawDetectedMarkers(display, corners_list, ids)

        try:
            diamond_corners, diamond_ids = cv2.aruco.detectCharucoDiamond(
                gray, corners_list, ids, SQUAREMARKERRATIO
            )
        except Exception:
            diamond_corners, diamond_ids = None, None

        if diamond_ids is not None and len(diamond_ids) > 0:
            for i in range(len(diamond_ids)):
                idtuple = tuple(int(x) for x in np.array(diamond_ids[i]).flatten())

                if idtuple == TARGET_IDS:
                    found_target = True

                    pts = np.array(diamond_corners[i], dtype=np.float32).reshape(4, 2)
                    pts_int = pts.astype(int)

                    cv2.polylines(display, [pts_int], True, (0, 255, 0), 3)

                    center = np.mean(pts, axis=0).astype(int)
                    cv2.circle(display, tuple(center), 6, (0, 0, 255), -1)

                    cv2.putText(
                        display,
                        "TARGET (16,17,18,19)",
                        (center[0] + 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

    if found_target:
        cv2.putText(display, "FOUND", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        cv2.putText(display, "NOT FOUND", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Single ChArUco Diamond Detect", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
