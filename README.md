# Vision-Based Cable-Driven Parallel Robot (CDPR) Control Stack

This repository contains a decoupled, cross-platform software framework for a 4-cable, 3-Degree-of-Freedom (DOF) translational Cable-Driven Parallel Robot (CDPR). Originally designed for automated surface water cleaning, this codebase implements a full "Vision-in-the-Loop" closed-loop system using:

- **Python (OpenCV):** Real-time end-effector tracking via ChArUco diamond fiducial markers, stabilized by Kalman Filtering.
- **MATLAB:** Newton-Euler feedforward dynamics modeling, safe Proportional-Derivative (PD) feedback tracking, and a Minimum-Variance Quadratic Programming (QP) tension optimization engine.
- **Arduino (C++):** Low-level, multi-axis synchronous stepper motor actuation using non-blocking serial communication.

---

## 🏗️ System Architecture & Data Flow

+-----------------------------------+

+-----------------------------------+
|
v (OpenCV / IPPE PnP)
+-----------------------------------+

| Overhead Video Stream | +-----------------------------------+
|
v (Telemetry over UDP)
+-----------------------------------+

| Python State Estimation | +-----------------------------------+
|
v (Serial/COM)
+-----------------------------------+

| MATLAB Control Loop Engine | +-----------------------------------+
| Arduino Microcontroller Node |
---

## 📂 Codebase Directory Breakdown

### 🎯 1. Calibration & Setup Utilities (`/vision/calibration`)

- `capture_charuco_checkerboard.py`: Streamlines capturing high-quality images from a webcam or external camera. It checks for a minimum corner detection threshold before allowing an image to save, ensuring data quality.
- `camera_calibration.py`: Processes saved calibration images to compute the camera's intrinsic matrix and distortion coefficients, writing them to a portably loaded `phonecameracalib.npz` file.
- `test_detect_single_marker.py`: A diagnostic tool to verify your camera index, resolution, and successful isolation of target ChArUco diamond IDs.

### 🎥 2. Real-Time Tracking Engines (`/vision/tracking`)

- `CV_2.py`: The production tracking node for a static-anchor system. It performs sub-pixel marker localization, disambiguates pose inversion via reprojection error screening, applies Exponential Moving Average (EMA) and Kalman filtering, and streams spatial telemetry via UDP.
- `reconf.py`: An extended variant engineered for **reconfigurable anchor setups**. It calculates the operational tension-positive workspace polygon in real-time, detecting if dynamically adjusted anchors put target points out of bounds.

### 💻 3. Computation & Trajectory Generation (`/control`)

- `MATLAB.m`: The master control loop. It orchestrates quintic S-curve tracking profiles, calculates the geometric 3x4 force structure matrix, optimizes cable tensions via QP to prevent slack, incorporates safety limitations (Z-gain elimination, error dead-bands), and issues hardware commands.

### 🔌 4. Low-Level Actuation (`/firmware`)

- `stepper_control.ino` (Arduino Code): Listens to the Serial port using a custom, non-blocking character parsing buffer. Receives differential step adjustments `(s1,s2,s3,s4)` and uses the `AccelStepper` library to run four motors simultaneously without stalling execution loops.

---

## ⚙️ Prerequisites & Environment Setup

### 🐍 Python Environment

Install dependencies using `pip`:

```bash
pip install opencv-contrib-python numpy
```

**Note:** Make sure to install `opencv-contrib-python` rather than the base `opencv-python` package, as the ArUco/ChArUco modules reside in the contrib repository.

### 🛠️ MATLAB Configuration

- **Optimization Toolbox:** Required for the `quadprog` optimization routine.
- **Instrument Control Toolbox:** Required to open `udp` client sockets and interface with serial ports (`serialport`).

---

## 🔧 Porting Guide: Customizing for a Completely New Setup

If you are cloning this repository to build your own CDPR from scratch with a different frame size, different anchors, or different motors, follow these adjustment steps:

### 1. Match Your ChArUco Calibration Target

If your physical printed calibration board changes layout, modify the definitions at the top of **both** `capture_charuco_images.py` and `calibrate_charuco.py`:

```python
SQUARES_X = 7              # Number of squares along the X axis
SQUARES_Y = 5              # Number of squares along the Y axis
SQUARE_LENGTH = 0.040      # Length of a chessboard square side in meters
MARKER_LENGTH = 0.028      # Length of an inner ArUco marker side in meters
ARUCO_DICT = cv2.aruco.DICT_4X4_50  # Must match printed dictionary type
```

### 2. Configure Python Vision Parameters (`CV_2.py` / `reconf.py`)

Open the tracking script and adjust your network configurations, geometric dimensions, and mounting offsets:

```python
CAM_INDEX = 1                 # 0 for integrated webcam, 1+ for USB/virtual camera links
MATLAB_IP = "192.168.137.179" # The IPv4 address of the computer running MATLAB
MATLAB_PORT = 5005            # Destination port for UDP packets

# Match these exactly to your physical end-effector assembly (in centimeters)
SQUARE_LENGTH_CM = 4.1
MARKER_LENGTH_CM = 2.4

# Offsets to map the optical center of the target marker to the actual CDPR end-effector pivot point
EE_OFFSET_X = 1.7
EE_OFFSET_Y = 0.5
EE_OFFSET_Z = 9.0

# Offsets mapping your camera coordinates to the origin corner of your physical tank/workspace frame
ORIGIN_OFFSET_X = 5.5
ORIGIN_OFFSET_Y = 6.6
ORIGIN_OFFSET_Z = 3.2
```

### 3. Configure MATLAB Physical Constants (`MATLAB.m`)

Update the system kinematics, mass profiles, and hardware boundaries inside `MATLAB.m`:

```matlab
% --- Physical Dimensions & Geometry ---
% Define the 3D coordinates (x,y,z) of your 4 anchor points on your frame:
A1 = ; 
A2 = ;
% ... Define A3, A4 to reflect your custom workspace width, length, and height.

% --- Mass & Mechanical Properties ---
m = 0.350;        % Mass of your end-effector assembly in Kilograms
r_drum = 1.25;    % Radius of your motor spool pulleys in centimeters
steps_per_rev = 1600; % Micro-stepping configurations configured on your motor drivers

% --- Tension Stabilization Floor & Ceiling ---
Tmin = 50;        % Lower tension floor (in g*cm/s^2) to prevent lines going slack
Tmax = 5000;      % Maximum safe tension ceiling to avoid snapping strings or stalling drivers
```

### 4. Configure Arduino Pin Assignments

If your CNC shield or wiring harness connects differently, reassign the STEP and DIRECTION digital pins in the instantiation block in your Arduino file:

```cpp
// AccelStepper driver instantiation (Interface Type = 1 means external step/direction driver)
#define MOTOR_INTERFACE_TYPE 1

AccelStepper stepper1(MOTOR_INTERFACE_TYPE, 2, 3); // (Type, Step Pin, Direction Pin)
AccelStepper stepper2(MOTOR_INTERFACE_TYPE, 4, 5);
AccelStepper stepper3(MOTOR_INTERFACE_TYPE, 6, 7);
AccelStepper stepper4(MOTOR_INTERFACE_TYPE, 8, 9);
```

---

## 🚀 Step-by-Step Execution Sequence

Execute the sub-modules systematically in this precise sequence to establish clean operational loops:

### Step 1: Intrinsic Camera Calibration

1. Run `python capture_charuco_images.py`. Move your board through a variety of angles, distances, and focal zones. Press `s` to save frames when the display confirms `READY`.
2. Run `python calibrate_charuco.py`. This parses the captured frame matrix and saves a local `phonecameracalib.npz` asset.

### Step 2: Flash Low-Level Firmware

1. Open the `.ino` sketch in your Arduino IDE.
2. Compile and upload the code to your microcontroller.
3. Open the Serial Monitor at `115200` baud. Ensure you see the startup sequence output: `Arduino Ready`. **Close the Serial Monitor before initializing MATLAB** to prevent port access locking errors.

### Step 3: Run the Vision Telemetry Server

1. Execute the primary state estimator script:

```bash
python CV_2.py
```

2. The vision tracker will boot, verify calibration parameters, acquire the video device stream, and continuously await tracking verification.

### Step 4: Run the Control Core

1. Open `MATLAB.m` within your MATLAB environment.
2. Verify that the configured `serialport` address matches your active microcontroller port assignment (e.g., `'COM3'` or `'/dev/ttyUSB0'`).
3. Run the script. MATLAB will automatically:
   - Establish connection hooks with the Arduino.
   - Bind to the local UDP port to ingest Python tracking matrices.
   - Calculate trajectory parameters and begin dynamic tracking execution.

---

## 🛡️ Built-in Safety Protections

To protect your hardware when trying new tracking setups or testing aggressive trajectories, the system includes several safety constraints:

- **Z-Axis Gain Zeroing:** Vertical position tracking feedback coefficients are set to zero. Gravity compensation is calculated purely via the feedforward mathematical model, preventing unstable vertical oscillations caused by out-of-plane camera latency.
- **Feedback Saturation Caps:** Proportional-Derivative corrections cannot account for more than **15%** of the baseline feedforward wrench weight, isolating the system from runaway acceleration spikes.
- **Vision-Loss Timeout Fallback:** If the tracking framework drops marker recognition frames or encounters network drops exceeding `0.5` seconds, MATLAB temporarily drops out of closed-loop configuration and shifts to pure feedforward open-loop operation until visibility returns.
