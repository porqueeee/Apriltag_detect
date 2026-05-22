from djitellopy import Tello
import cv2
import numpy as np
# =========================
# LINE FOLLOWER
# =========================

def follow(frame):
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Yellow range
    lower = np.array([80, 80, 80])
    upper = np.array([110, 255, 255])

    # Mask
    mask = cv2.inRange(hsv, lower, upper)

    # Noise reduction
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw detected line
    if contours:

        c = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(c)

        if area > 500:

            x, y, w, h = cv2.boundingRect(c)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Center
            cx = x + w // 2
            cy = y + h // 2

            cv2.circle(
                frame,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            print("Yellow line detected")
    return mask

# =========================
# TELLO SETUP
# =========================

tello = Tello()
tello.connect()
print("Battery:", tello.get_battery(), "%")
# Start camera
tello.streamon()
frame_read = tello.get_frame_read()

# =========================
# YELLOW DETECTION
# =========================

while True:

    frame = frame_read.frame

    if frame is None:
        continue

    mask= follow(frame)

    # Show windows
    cv2.imshow("Tello Camera", frame)
    cv2.imshow("Yellow Mask", mask)

    # Exit with Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================

tello.streamoff()

cv2.destroyAllWindows()