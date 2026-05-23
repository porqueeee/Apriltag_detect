import numpy as np
import matplotlib.pyplot as plt
import cv2
from djitellopy import Tello


ap0=np.array([[0, 0], [0, 7], [1, 1], [1, 2], [1, 3], [1, 6], [2, 1]])
ap1=np.array([[0, 0], [0, 7], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5] ])
ap2=np.array([[0, 0], [0, 7], [1, 1], [1, 3], [1, 4], [2, 1], [2, 3]])
ap3=np.array([[0, 0], [0, 7], [1, 1], [1, 4], [2, 2], [2, 4], [2, 6]])
ap4=np.array([[0, 0], [0, 7], [1, 1], [1, 3], [2, 1], [2, 2], [2, 5]])
ap5=np.array([[0, 0], [0, 7], [1, 4], [1, 5], [1, 6], [2, 2], [2, 3]])
ap6=np.array([[0, 0], [0, 7], [1, 3], [1, 5], [2, 2], [2, 3], [2, 4]])
ap7=np.array([[0, 0], [0, 7], [1, 1], [1, 2], [1, 3], [1, 4], [2, 1]])
ap8=np.array([[0, 0], [0, 7], [1, 1], [1, 3], [1, 4], [1, 5], [2, 2]])
ap9=np.array([[0, 0], [0, 7], [1, 2], [1, 3], [1, 4], [1, 5], [2, 1]])
ap10=np.array([[0, 0], [0, 7], [1, 3], [1, 4], [2, 1], [2, 2], [2, 3]])
ap11=np.array([[0, 0], [0, 7], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]])

dictionary=np.concatenate(([ap0],[ap1],[ap2],[ap3],[ap4],[ap5],[ap6],[ap7],[ap8],[ap9],[ap10],[ap11]))
print(len(dictionary))

from djitellopy import Tello
import cv2
import numpy as np
import time
import pupil_apriltags as apriltag

# ====================================
# TELLO SETUP
# ====================================

tello = Tello()

print("Connecting...")
tello.connect()

battery = tello.get_battery()
print("Battery:", battery, "%")

# Start camera
tello.streamon()

frame_read = tello.get_frame_read()

# ====================================
# APRILTAG DETECTOR
# ====================================

detector = apriltag.Detector()

# ====================================
# TAKEOFF
# ====================================

print("Takeoff...")
tello.takeoff()

time.sleep(2)

print("Ascending...")

start_up = time.time()

while time.time() - start_up < 1.2:

    # up-down velocity
    tello.send_rc_control(0, 0, 10, 0)

    time.sleep(0.05)

# Stop movement
tello.send_rc_control(0, 0, 0, 0)

# ====================================
# PID HOVER FOR 10 SECONDS
# ====================================

print("Hovering for 10 seconds...")

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

while time.time() - hover_start < 10:

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

# ====================================
# APRILTAG DETECTION
# ====================================

print("Searching for AprilTags...")

while True:

    frame = frame_read.frame

    if frame is None:
        continue

    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect tags
    tags = detector.detect(gray)

    # Draw detections
    for tag in tags:

        corners = tag.corners.astype(int)

        for i in range(4):

            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i + 1) % 4])

            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        center = (int(tag.center[0]), int(tag.center[1]))

        cv2.circle(frame, center, 5, (0, 0, 255), -1)

        tag_id = tag.tag_id

        cv2.putText(
            frame,
            f"ID: {tag_id}",
            (center[0] - 20, center[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        print("AprilTag detected! ID:", tag_id)

    # Keep drone hovering
    tello.send_rc_control(0, 0, 0, 0)

    cv2.imshow("AprilTag Detection", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ====================================
# LAND
# ====================================

print("Landing...")

tello.send_rc_control(0, 0, 0, 0)

tello.land()

tello.streamoff()

cv2.destroyAllWindows()

tello.end()

print("Drone stopped")