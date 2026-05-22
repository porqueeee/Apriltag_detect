import numpy as np
import matplotlib.pyplot as plt
import cv2

def show(img, title="Imagen", cmap =None, size =5):
    plt.figure(figsize=(size,size))
    if len(img.shape) == 2:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('on')
    plt.show()
                   
canvas= np.zeros((400,400,3),dtype=np.uint8)


cv2.rectangle(canvas, (0,0), (400,400), (190,140,40), -1)
cv2.rectangle(canvas, (180,50), (220,400), (70,70,70), -1)
cv2.rectangle(canvas, (150,40), (250,252), (100,100,100), -1)
cv2.circle(canvas, (200,80), 28, (20,180,20), -1)
cv2.circle(canvas, (200,145), 28, (20,180,180), -1)
cv2.circle(canvas, (200,210), 28, (20,20,180), -1)

cv2.circle(canvas, (134,50), 10, (180,60,200), -1)
pts= np.array([[150,40],[120,44],[140,70]], np.int32).reshape(-1,1,2)
cv2.polylines(canvas, [pts], isClosed=True,color=(180,60,200), thickness=(8))
cv2.polylines(canvas, [pts], isClosed=True,color=(100,60,130), thickness=(3))

cv2.circle(canvas, (170,30), 10, (180,60,200), -1)
pts= np.array([[150,40],[170,20],[185,40]], np.int32).reshape(-1,1,2)
cv2.polylines(canvas, [pts], isClosed=True,color=(180,60,200), thickness=(8))
cv2.polylines(canvas, [pts], isClosed=True,color=(100,60,130), thickness=(3))


cv2.circle(canvas, (150,40), 10, (180,60,200), -1)
cv2.circle(canvas, (150,40), 11, (100,60,130), 2)


fuente=cv2.FONT_HERSHEY_SCRIPT_COMPLEX
cv2.putText(canvas, 'semaforo', (250,380), fuente, 1, (200,200,200),2,cv2.LINE_AA)

show(canvas,"Semáforo")