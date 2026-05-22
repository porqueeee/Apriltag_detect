#Código usado el viernes

import cv2
import numpy as np
from djitellopy import Tello
import time
import statistics

# Inicializar el dron Tello
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

# Iniciar la transmisión de video
tello.streamon()
count = 0
veredict = ['none'] *15
tello.set_speed(15)

time.sleep(1) 
tello.takeoff()
time.sleep(1) 
tello.move_down(20)
#tello.move_forward(20)

def detect_color(color):
    B, G, R = color
    Y=abs(R-G)
    if (R > 120) and (G > 120) and (B < 120) and (Y<100) :
        return 'yellow'
    elif (G > 150) and (B > 150) and (R > 150) and (Y<100):
        return 'white'
    elif (G < 150) and (B < 150) and (R < 150) and (Y<100):
        return 'black'
    elif (G < 150) and (B < 150) and (R > 150) and (Y>50):
        return 'red'
    else:
        return 'green'


def detect_shape_and_color(frame):
    shapes_colors = ''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    #_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width, _ = frame.shape
    radius = 80  # puedes ajustar el radio
    center_x = width // 2
    center_y = height // 2
    circle = np.zeros(rgb.shape[:2], dtype=np.uint8)
    cv2.circle(circle, (center_x, center_y), radius, 255, -1)
    _, thresh = cv2.threshold(circle, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:
        area = cv2.contourArea(contour, False)
        if 10000 < area < 48000:
            approx = cv2.approxPolyDP(contour, 0.035 * cv2.arcLength(contour, True), True)
            shape = 'unknown'
            convex = cv2.isContourConvex(approx)
            sides = len(approx)
            if sides == 3:
                shape = 'triangle'
            if sides == 4:
                shape = 'square'
            elif sides == 5:
                shape = 'pentagon'
            elif sides >= 7 and convex and sides < 40:
                shape = 'circle'
            #elif sides >= 7 and not convex and sides < 30:
                #shape = 'star'

            mask = np.zeros(rgb.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_color = cv2.mean(rgb, mask=mask)[:3]
            detected_color = detect_color(mean_color)

            
            shapes_colors = detected_color
            cv2.drawContours(rgb, [contour], 0, (0, 255, 0), 2)
            break  # salir tras detectar la primera forma válida

    return shapes_colors, rgb

end=0
change=0

# ...código inicial intacto...

# Dentro del bucle try:
try:
    while True:
        count = count % 15
        frame = tello.get_frame_read().frame
        shapes_colors, processed_frame = detect_shape_and_color(frame)

        veredict[count] = shapes_colors if shapes_colors != '' else 'none'

        try:
            final = statistics.mode([v for v in veredict if v != 'none'])
        except statistics.StatisticsError:
            final = 'none'

        cv2.putText(processed_frame, f"{final}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Tello Camera", processed_frame)

        # --- COMPORTAMIENTO SEGÚN COLOR ---
        if final == 'black' and count==0:
            end += 1
            tello.send_rc_control(15, 0, 0, 0)
            time.sleep(1)
            tello.send_rc_control(0, 0, 0, 0)
            if end > 5:
                break

        elif final == "yellow" or final == 'green' and count==0:
            tello.send_rc_control(15, 0, 0, 0)
            time.sleep(1)
            tello.send_rc_control(0, 0, 0, 0)

        elif final == 'red' and count==0:
            tello.rotate_counter_clockwise(90)

        elif final == 'white' and count==0:
            tello.rotate_clockwise(90)
        elif count!=0:
            tello.send_rc_control(0, 0, 0, 0)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrumpido manualmente.")

finally:
    tello.send_rc_control(0, 0, 0, 0)  # asegurarse de detener el movimiento
    tello.land()
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()