from djitellopy import Tello
import cv2
import time
import pupil_apriltags as apriltag
import numpy as np

# ====================================
# TELLO SETUP (SOLO CÁMARA)
# ====================================

tello = Tello()

print("Connecting...")
tello.connect()

print("Battery:", tello.get_battery(), "%")


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
# START CAMERA
# ====================================

tello.streamon()
time.sleep(2)

frame_read = tello.get_frame_read()


# ====================================
# PID HOVER FOR 7 SECONDS
# ====================================

print("Hovering for 7 seconds...")

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

while time.time() - hover_start < 7:

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

tello.send_rc_control(0, 0, 0, 0)

#Baja a 25 centímetros para mejor detección
height=tello.get_height()
while height > 25:
    height=tello.get_height()
    print(height)
    tello.send_rc_control(0, 0, -10, 0)
    time.sleep(0.1)


# ====================================
# APRILTAG DETECTOR
# ====================================

detector = apriltag.Detector(
    families="tag25h9"
)

VALID_IDS = list(range(12))

# Control para no spamear mensajes
last_action_time = 0
cooldown = 2  # segundos

# ====================================
# FUNCTIONS
# ====================================

def action(state):
    match state:
        case "Fin del vuelo":
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
        case "Avanzar":
            tello.send_rc_control(0, 10, 0, 0)
        case "Giro 360":
            tello.rotate_clockwise(360)
        case "Giro 180":
            tello.rotate_clockwise(180)
        case "Giro 90":
            tello.rotate_clockwise(90)
        case "Girar derecha":
            tello.rotate_clockwise(90)
        case "Girar izquierda":
            tello.rotate_counter_clockwise(90)
        case _:
            tello.send_rc_control(0, 10, 0, 0)


# ====================================
# DETECTION LOOP
# ====================================

print("Detecting AprilTags 0-11...")

while True:
    tello.send_rc_control(0, 10, 0, 0)
    frame = frame_read.frame
    #Corrige el espejeado
    frame = cv2.flip(frame, 0)
    frame=frame[280:655,171:548,:].copy()

    if frame is None:
        continue

    #mejor mantener la resolución orignal
    #frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    tags = detector.detect(gray)

    for tag in tags:

        tag_id = tag.tag_id

        if tag_id not in VALID_IDS:
            continue
        
        tello.send_rc_control(0, 0, 0, 0)

        corners = tag.corners.astype(int)

        # Dibujar cuadro
        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i + 1) % 4])

            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Centro
        center = (
            int(tag.center[0]),
            int(tag.center[1])
        )

        cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # ====================================
        # ACCIONES (SOLO TEXTO)
        # ====================================

        current_time = time.time()

        if current_time - last_action_time < cooldown:
            continue

        action_text = ""

        if tag_id == 1 or tag_id == 9:
            action_text = "Fin del vuelo"

        elif tag_id == 2 or tag_id == 11:
            action_text = "Avanzar"

        elif tag_id == 3 or tag_id == 0:
            action_text = "Giro 360"

        elif tag_id == 4:
            action_text = "Giro 180"

        elif tag_id == 5 or tag_id == 8:
            action_text = "Girar izquierda"

        elif tag_id == 6:
            action_text = "Giro 90"

        elif tag_id == 7 or tag_id == 10:
            action_text = "Girar derecha"

        action(action_text)

        print(f"ID {tag_id}: {action_text}")

        # Mostrar acción en pantalla
        cv2.putText(
            frame,
            action_text,
            (center[0] - 50, center[1] + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        last_action_time = current_time

        # Mostrar ID también
        cv2.putText(
            frame,
            f"ID: {tag_id}",
            (center[0] - 20, center[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    cv2.imshow("AprilTag Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

# ====================================
# CLOSE EVERYTHING
# ====================================
tello.send_rc_control(0, 0, 0, 0)
tello.land()
tello.streamoff()
cv2.destroyAllWindows()
tello.end()

print("Program closed")