import cv2
import statistics
import numpy as np


# Open the default camera
cam = cv2.VideoCapture(2)
cam.set(3 , 640) # width    
cam.set(4 , 480) # height       
cam.set(10,100) #brightness     min: 0   , max: 255 , increment:1  
cam.set(11, 106) # contrast       min: 0   , max: 255 , increment:1     
cam.set(12, 109) # saturation     min: 0   , max: 255 , increment:1      
cam.set(14, 35) # gain           min: 0   , max: 127 , increment:1
cam.set(15, -6) # exposure       min: -7  , max: -1  , increment:1
cam.set(17, 4500) # white_balance  min: 4000, max: 7000, increment:1
cam.set(28, 0)
veredict = ['none'] * 15
count = 0
_, imageFrame = cam.read() 


#definir funciones

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
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            break  # salir tras detectar la primera forma válida

    return shapes_colors, rgb




# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()

    count = count % 15
    shapes_colors, processed_frame = detect_shape_and_color(frame)

    veredict[count] = shapes_colors if shapes_colors != '' else 'none'


    try:
        final = statistics.mode([v for v in veredict if v != 'none'])
    except statistics.StatisticsError:
        final = 'none'




    # Write the frame to the output file
    out.write(frame)
    # Display the captured frame
    cv2.imshow('Camera', frame)
    print(shapes_colors)
    cv2.putText(imageFrame, "shapes_colors", (0,0 ), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255, 0, 0))

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break
        print("Interrumpido manualmente.")

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()