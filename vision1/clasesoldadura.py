import cv2
import numpy as np
import matplotlib.pyplot as plt

def analizar_soldadura(ruta_imagen):
    
    # 1. Cargar la imagen
    img = cv2.imread(ruta_imagen)
    if img is None:
        print("No se pudo cargar la imagen.")
        return

    # Escala de simulación: ¿Cuántos píxeles hay en 1 mm? 
    # (Este valor depende de la distancia de la cámara)
    PIXELS_PER_MM = 10 
    ESTANDAR_ANCHO_MM = 10
    ESTANDAR_LARGO_MM = 80 # 10 cm

    # 2. Pre-procesamiento: Escala de grises y Blur
    # El desenfoque ayuda a eliminar el ruido metálico y reflejos
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 3. Threshold (Umbralización)
    # Usamos Otsu para encontrar el umbral óptimo automáticamente
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Operaciones Morfológicas
    # 'Opening' para eliminar pequeños puntos blancos (ruido)
    # 'Closing' para cerrar huecos dentro del cordón
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Detección de Bordes (Canny)
    edges = cv2.Canny(closing, 50, 150)

    # 6. Detección de Contornos
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Suponemos que el contorno más grande es el cordón de soldadura
        cnt = max(contours, key=cv2.contourArea)

        # Obtener el rectángulo rotado mínimo que contiene el contorno
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Dimensiones en píxeles
        (x, y), (w, h), angle = rect
        
        # Identificar cuál es el largo y cuál es el ancho
        ancho_px = min(w, h)
        largo_px = max(w, h)

        # Convertir a unidades métricas
        ancho_mm = ancho_px / PIXELS_PER_MM
        largo_mm = largo_px / PIXELS_PER_MM

        # 7. Lógica de Decisión (Pasa / No Pasa)
        # Tolerancia del 10% ejemplo
        pasa_ancho = ancho_mm >= ESTANDAR_ANCHO_MM
        pasa_largo = largo_mm >= ESTANDAR_LARGO_MM

        resultado = "PASA" if (pasa_ancho and pasa_largo) else "NO PASA"

        # Dibujar resultados en la imagen
        cv2.drawContours(img, [box], 0, (0, 255, 0) if resultado == "PASA" else (0, 0, 255), 3)
        cv2.putText(img, f"Ancho: {ancho_mm:.2f}mm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, f"Largo: {largo_mm/10:.2f}cm", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, f"Status: {resultado}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if resultado == "PASA" else (0, 0, 255), 3)

        # Mostrar imágenes del proceso (opcional)
        #cv2.imshow("Threshold", thresh)
        #cv2.imshow('Morph',closing)
        #cv2.imshow('Result',edges)
        #cv2.imshow("Resultado Final", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        plt.figure(figsize=(10,10))
        f, grid = plt.subplots(2,2) 
        grid[0,0].axis('off')
        grid[0,1].axis('off')
        grid[1,0].axis('off')
        grid[1,1].axis('off')
        grid[0,0].imshow(thresh)
        grid[0,1].imshow(closing)
        grid[1,0].imshow(edges)
        grid[1,1].imshow(img)


        print(f"Resultados:\nAncho: {ancho_mm:.2f} mm\nLargo: {largo_mm:.2f} mm\nEstatus: {resultado}")
    else:
        print("No se detectó ningún cordón de soldadura.")
    #return ancho_mm, largo_mm, pasa_ancho, pasa_largo, ESTANDAR_ANCHO_MM, ESTANDAR_LARGO_MM
    return

# Para ejecutar:
#ancho, largo, pasa_a, pasa_l, e_a, e_l = analizar_soldadura('soldadura.jpg')
analizar_soldadura('C:/Users/valef/Downloads/vision1/Archivos del curso/soldadura1.jpg')