""" Día 5 - Taller de Python ~ OpenCV
Angie Contreras ~ Club de Robótica

HSV """

# -- importar opencv y numpy--
import cv2 as cv
import numpy as np

""" Actividad 1. HSV """

def nothing(x):
    pass

# Crear ventana y trackbars para rangos mínimos y máximos de HSV
cv.namedWindow("Panel", cv.WINDOW_NORMAL)
cv.createTrackbar("H Min", "Panel", 0, 179, nothing)
cv.createTrackbar("S Min", "Panel", 0, 255, nothing)
cv.createTrackbar("V Min", "Panel", 0, 255, nothing)
cv.createTrackbar("H Max", "Panel", 179, 179, nothing)
cv.createTrackbar("S Max", "Panel", 255, 255, nothing)
cv.createTrackbar("V Max", "Panel", 255, 255, nothing)

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    h_min = cv.getTrackbarPos("H Min", "Panel")
    s_min = cv.getTrackbarPos("S Min", "Panel")
    v_min = cv.getTrackbarPos("V Min", "Panel")
    h_max = cv.getTrackbarPos("H Max", "Panel")
    s_max = cv.getTrackbarPos("S Max", "Panel")
    v_max = cv.getTrackbarPos("V Max", "Panel")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv.inRange(hsv, lower, upper)
    result = cv.bitwise_and(frame, frame, mask=mask)
    mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)  # Para mostrar la máscara en color

    # Unir las imágenes horizontalmente
    panel = np.hstack((frame, mask_bgr, result))

    cv.imshow("Panel", panel)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()