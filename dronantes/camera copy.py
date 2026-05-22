import cv2
import statistics
import numpy as np


# Open the default camera
cam = cv2.VideoCapture(0)
cam.set(3 , 640) # width    
cam.set(4 , 480) # height       
cam.set(10,60) #brightness     min: 0   , max: 255 , increment:1  
cam.set(11, 106) # contrast       min: 0   , max: 255 , increment:1     
cam.set(12, 200) # saturation     min: 0   , max: 255 , increment:1      
cam.set(14, 35) # gain           min: 0   , max: 127 , increment:1
cam.set(15, -7) # exposure       min: -7  , max: -1  , increment:1
cam.set(17, 4500) # white_balance  min: 4000, max: 7000, increment:1
cam.set(28, 0)

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
        return 'pink'
    else:
        return 'green'
    
def detect_shape_and_color(frame):
    shapes_colors = ''
    rgb = frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
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

            shapes_colors = shape, detected_color
            cv2.drawContours(rgb, [contour], 0, (0, 255, 0), 2)
            cv2.putText(rgb, f"{shape+ detected_color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            break  # salir tras detectar la primera forma válida

    return shapes_colors, rgb



while True:
    ret, frame = cam.read()
    shapes_colors, processed_frame = detect_shape_and_color(frame)
    cv2.imshow('Camera', processed_frame)
    print(shapes_colors)
    cv2.putText(processed_frame, "shapes_colors", (0,0 ), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255, 0, 0))

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break
        print("Interrumpido manualmente.")

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()