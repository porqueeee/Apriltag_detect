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
    plt.axis('off')
    plt.show()

h,w =300,500
img =np.zeros((h,w), dtype = np.uint8)

for x in range(w):
    img[:,x] =int(255*x / (w-1))

cv2.rectangle(img, (40,40), (150,150), 220, -1)
cv2.circle(img, (400,80), 40, 20, -1)

fuente=cv2.FONT_HERSHEY_PLAIN
cv2.putText(img, 'THRESH', (100,100), fuente, 5, (200,200,200) ,2,cv2.LINE_AA)
cv2.putText(img, 'THRESH', (100,250), fuente, 5, (200,200,200) ,2,cv2.LINE_AA)

corte, thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)

show(thresh, 'Degradado', cmap= 'gray', size=7)