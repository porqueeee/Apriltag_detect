import numpy as np
import matplotlib.pyplot as plt
import cv2

img= cv2.imread("C:/Users/valef/Downloads/vision1/Archivos del curso/ladrillos.jpg").astype(np.float32)/255
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#gamma=1/4
#gamma_img=np.power(img,gamma)


fuente=cv2.FONT_HERSHEY_PLAIN
cv2.putText(img, 'THRESH', (10,350), fuente, 5, (0,60,60) ,3)

img_blur=cv2.blur(img,(7,7))

plt.figure()
plt.imshow(img_blur)

