import cv2
import numpy as np
import time
from djitellopy import tello  # Matches second script's instantiation pattern
from scipy import stats

# APRILTAG DICTIONARY ( falta agregar las rotaciones :( )

veredict = []
count = 0

ap0 = np.array([[0, 0], [0, 7], [1, 1], [1, 2], [1, 3], [1, 6], [2, 1]])
ap1 = np.array([[0, 0], [0, 7], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
ap2 = np.array([[0, 0], [0, 7], [1, 1], [1, 3], [1, 4], [2, 1], [2, 3]])
ap3 = np.array([[0, 0], [0, 7], [1, 1], [1, 4], [2, 2], [2, 4], [2, 6]])
ap4 = np.array([[0, 0], [0, 7], [1, 1], [1, 3], [2, 1], [2, 2], [2, 5]])
ap5 = np.array([[0, 0], [0, 7], [1, 4], [1, 5], [1, 6], [2, 2], [2, 3]])
ap6 = np.array([[0, 0], [0, 7], [1, 3], [1, 5], [2, 2], [2, 3], [2, 4]])
ap7 = np.array([[0, 0], [0, 7], [1, 1], [1, 2], [1, 3], [1, 4], [2, 1]])
ap8 = np.array([[0, 0], [0, 7], [1, 1], [1, 3], [1, 4], [1, 5], [2, 2]])
ap9 = np.array([[0, 0], [0, 7], [1, 2], [1, 3], [1, 4], [1, 5], [2, 1]])
ap10 = np.array([[0, 0], [0, 7], [1, 3], [1, 4], [2, 1], [2, 2], [2, 3]])
ap11 = np.array([[0, 0], [0, 7], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]])

dictionary = np.concatenate(([ap0], [ap1], [ap2], [ap3], [ap4], [ap5], [ap6], [ap7], [ap8], [ap9], [ap10], [ap11]))

tolerancia = 10

# DRONE INITIALIZATION & STARTUP

print("Connecting...")
tello = tello.Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

tello.streamon()
frame_read = tello.get_frame_read()
time.sleep(2)  # Give OpenCV background thread a moment to initialize the camera pipeline

print("Takeoff...")
tello.takeoff()
time.sleep(2)

print("Ascending...")
start_up = time.time()
while time.time() - start_up < 1.2:
    tello.send_rc_control(0, 0, 10, 0)
    time.sleep(0.05)
tello.send_rc_control(0, 0, 0, 0)

# APRILTAG DETECTION
def apriltag(tablero):  
    tablero = tablero[280:655, 171:548, :].copy()
    tablero = cv2.cvtColor(tablero, cv2.COLOR_BGR2RGB)
    tablero_gris = cv2.cvtColor(tablero, cv2.COLOR_BGR2GRAY)
    tablero_gris_float = np.float32(tablero_gris)
    
    destino = cv2.cornerHarris(tablero_gris_float, 2, 7, 0.04)
    destino = cv2.dilate(destino, None)

    ret, destino = cv2.threshold(destino, 0.01*destino.max(), 255, 0)
    destino = np.uint8(destino)
    ret, labels, stats_cc, centroids = cv2.connectedComponentsWithStats(destino)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    
    # Catching edge case where no corners are tracked to prevent cv2 crashes
    if len(centroids) == 0:
        return [], tablero
        
    corners = cv2.cornerSubPix(tablero_gris_float, np.float32(centroids), (5, 5), (-1, -1), criteria)
    corners = corners[corners[:, 0].argsort()]

    for i in range(len(corners)):
        cv2.drawMarker(tablero, ((int(corners[i][0])), (int(corners[i][1]))), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

    gridlist = []
    organizer = [corners[0]]

    for i in range(1, len(corners)):
        if abs(corners[i][0]-corners[i-1][0]) <= tolerancia and corners[i][0] != corners[i-1][0]:
            organizer.append(corners[i])
        elif corners[i][0] != corners[i-1][0]:
            organizer = np.array(organizer)
            organizer = organizer[organizer[:, 1].argsort()]
            organizer = organizer.tolist()
            gridlist = gridlist + organizer
            organizer = [corners[i]]

    organizer = np.array(organizer)
    organizer = organizer[organizer[:, 1].argsort()]
    organizer = organizer.tolist()
    gridlist = gridlist + organizer

    length = len(gridlist) - 1
    side = int(abs(gridlist[0][1] - gridlist[-1][1]))
    fraction = int(side / 7)
    if fraction == 0:
        fraction = 1

    gridlist = np.array(gridlist)
    map_grid = gridlist / fraction

    id_tags = [[int(map_grid[0][0]), int(map_grid[0][1])]]
    for i in range(1, len(map_grid)):
        if int(map_grid[i][0]) == int(map_grid[i-1][0]) and int(map_grid[i][1]) == int(map_grid[i-1][1]):
            pass
        else:
            id_tags.append([int(map_grid[i][0]), int(map_grid[i][1])])

    cv2.rectangle(tablero, ((int(gridlist[0][0])), (int(gridlist[0][1]))), ((int(gridlist[0][0])+side), (int(gridlist[0][1])+side)), (0, 0, 255))
    cv2.rectangle(tablero, ((int(gridlist[0][0])+fraction), (int(gridlist[0][1])+fraction)), ((int(gridlist[-1][0])-fraction), (int(gridlist[-1][1]))-fraction), (255, 0, 0))
    
    return id_tags, tablero

# ====================================
# MAIN COMBINED CONTROL LOOP
# ====================================
print("Advancing and tracking features. Searching for AprilTag...")

# PID Variables from Script 1
Kp = 0.4
Ki = 0.0
Kd = 0.2
previous_error = 0
integral = 0
TARGET_X = 480 // 2

# Constant slow forward movement speed
FORWARD_SPEED = 10 

try:
    while True:
        frame = frame_read.frame
        if frame is None:
            continue

        # --- 1. STABILIZATION & FEATURE TRACKING LOGIC ---
        # Make a copy for stabilization pipeline processing
        stab_frame = cv2.resize(frame, (480, 360))
        gray = cv2.cvtColor(stab_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        corners_stab = cv2.goodFeaturesToTrack(gray, maxCorners=1, qualityLevel=0.01, minDistance=30)
        yaw_speed = 0

        if corners_stab is not None:
            # Replaced deprecated np.int0 handling contextually if needed; kept functionality pure
            corners_stab = np.int64(corners_stab) if hasattr(np, 'int64') else np.int0(corners_stab)
            for corner in corners_stab:
                x, y = corner.ravel()
                error = x - TARGET_X
                integral += error
                derivative = error - previous_error
                output = (Kp * error + Ki * integral + Kd * derivative)
                previous_error = error
                yaw_speed = int(max(min(output, 8), -8))
                cv2.circle(stab_frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow("Hover PID", stab_frame)

        # --- 2. CUSTOM APRILTAG DETECTION LOGIC ---
        # Raw frame conversion passed directly to your decoder function
        tag_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        id_detected, detect_view = apriltag(tag_frame)
        cv2.imshow("crop", detect_view)

        april = 100
        for i in range(len(dictionary)):
            if len(id_detected) >= 7:
                arr = np.absolute(id_detected[:7] - dictionary[i])
                sum_err = arr.sum()
                if sum_err < 3 and len(id_detected) < 35:
                    april = i
                    break

        if april != 100:
            print(f"Detected Code Match ID: {april}")
            veredict.append(april)
            count += 1
        else:
            print("Scanning...", end="\r")

        # --- 3. FLIGHT LOGIC DECISION MATRIX ---
        if count >= 10:
            # Fetch statistical mode of the matching windows
            moda_res = stats.mode(veredict)
            # Safe compatibility extraction for scipy versions
            moda = moda_res.mode[0] if hasattr(moda_res.mode, '__len__') else moda_res.mode
            
            print(f"\nFinal Verdict Tag Verified: {moda}. Initiating Safe Landing Sequence.")
            tello.send_rc_control(0, 0, 0, 0) # Stop all movement vectors instantly
            break # Break loop to process the mandatory clean landing sequence block below

        # --- 4. EXECUTE COMBINED RC MOVEMENT ---
        # Continues moving forward slowly at velocity (10) while dynamically applying PID yaw corrections
        tello.send_rc_control(0, FORWARD_SPEED, 0, yaw_speed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nUser Interrupted via Keyboard Interface.")
            break
            
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nInterrupted manually.")

finally:
    print("Landing...")
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()
    tello.end()
    print("Drone stopped safely.")