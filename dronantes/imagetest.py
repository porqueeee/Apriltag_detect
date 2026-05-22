import cv2
import numpy as np
from djitellopy import Tello
import time
import tkinter as tk


def detect_color(color):
    B=color[0]
    G=color[1]
    R=color[2]

    name='unknown'

    if (R>120) and (G<110) and (B<110):
        name='red'
    elif (G>100) and (B<120) and (R<120):
        name='green'

    return name


def detect_shape_and_color(frame):
    """
    Detecta formas geométricas y colores en el cuadro dado.
    Retorna una lista con las figuras y sus colores detectados.
    """
    shapes_colors = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    

    # Convertir a escala de grises y detectar bordes
    gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    for contour in contours:
        area=cv2.contourArea(contour, False)
        if (area > 10000 and area<48000): 
        # Detectar la forma geométrica
            approx = cv2.approxPolyDP(contour, 0.035 * cv2.arcLength(contour, True), True)
            shape = "unknown"
            convex= cv2.isContourConvex(approx)
            side= len(approx)
            if side == 4:
                shape = "square"
            elif side == 5:
                shape = "pentagon"
            elif side >= 7 and (convex==True) and side<40:
                shape = "circle"
            #elif len(approx) >= 7 and (convex=False):
                #shape= "star"

        # Detectar el color
            mask = np.zeros(rgb.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_color = cv2.mean(rgb, mask=mask)[:3]

            detected_color = detect_color(mean_color)


            if shape != "unknown": #and detected_color != "unknown":
                shapes_colors.append((shape, detected_color))
                cv2.drawContours(rgb, [contour], 0, (0, 255, 0), 2)
                cv2.putText(rgb, f"{shape} {detected_color}", tuple(approx[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return shapes_colors, rgb

# Procesar el video en tiempo real
try:
    while True:
        frame = cv2.imread("image.png", cv2.IMREAD_COLOR)
        shapes_colors, processed_frame = detect_shape_and_color(frame)  # Detectar figuras y colores

        # Mostrar las figuras y colores detectados en la ventana
        cv2.imshow("Image", processed_frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrumpido manualmente.")

finally:
    # Detener el dron y cerrar todo
    cv2.destroyAllWindows()