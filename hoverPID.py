from djitellopy import Tello
import cv2
import numpy as np
import time

tello = Tello()

print("Connecting...")
tello.connect()

battery = tello.get_battery()

print("Battery:", battery, "%")

# Start camera
tello.streamon()

frame_read = tello.get_frame_read()

# ====================================
# TAKEOFF
# ====================================

print("Takeoff...")

tello.takeoff()

time.sleep(2)

# ====================================
# ASCEND ~40 CM
# ====================================

print("Ascending...")

start_up = time.time()

while time.time() - start_up < 1.2:

    # up-down velocity
    tello.send_rc_control(0, 0, 10, 0)

    time.sleep(0.05)

# Stop movement
tello.send_rc_control(0, 0, 0, 0)

# ====================================
# PID HOVER FOR 20 SECONDS
# ====================================

print("Hovering for 20 seconds...")

# PID gains
Kp = 0.4
Ki = 0.0
Kd = 0.2

# PID variables
previous_error = 0
integral = 0

# Frame center
TARGET_X = 480 // 2

hover_start = time.time()

while time.time() - hover_start < 20:

    frame = frame_read.frame

    if frame is None:
        continue

    # Resize frame
    frame = cv2.resize(frame, (480, 360))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect features
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=1,
        qualityLevel=0.01,
        minDistance=30
    )

    yaw_speed = 0

    if corners is not None:

        corners = np.int0(corners)

        for corner in corners:

            x, y = corner.ravel()

            # PID error
            error = x - TARGET_X

            integral += error

            derivative = error - previous_error

            output = (
                Kp * error +
                Ki * integral +
                Kd * derivative
            )

            previous_error = error

            yaw_speed = int(
                max(min(output, 8), -8)
            )

            # Draw feature
            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

    # Keep hover stable
    tello.send_rc_control(
        0,          # left-right
        0,          # forward-back
        0,          # up-down
        yaw_speed   # yaw
    )

    cv2.imshow("Hover PID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.05)

print("Landing...")

tello.send_rc_control(0, 0, 0, 0)

tello.land()

tello.streamoff()
cv2.destroyAllWindows()
tello.end()
print("Drone stopped")