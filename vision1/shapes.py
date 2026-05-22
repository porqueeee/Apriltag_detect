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


cv2.rectangle(canvas, (50,60), (280,200), (255,255,0), -1)
cv2.rectangle(canvas, (70,80), (260,180), (0,0,0), -1)
cv2.line(canvas, (70,80), (260,180), (255,0,0), 3)
cv2.circle(canvas, (280,200), 100, (160,0,160), 5)
cv2.circle(canvas, (280,200), 80, (120,0,120), 5)
cv2.circle(canvas, (280,200), 40, (180,20,180), -1)

pts= np.array([[280,200],[240,320],[320,320]], np.int32).reshape(-1,1,2)
cv2.polylines(canvas, [pts], isClosed=True,color=(255,255,127), thickness=2)

show(canvas,"Figuras")