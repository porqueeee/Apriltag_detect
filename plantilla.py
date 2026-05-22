import numpy as np
import matplotlib.pyplot as plt
import cv2

img=cv2.imread("apt4.jpeg")
cara=cv2.imread("apriltag1.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, img= cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR )

copy=img.copy()
copy2=img.copy()
f, grid = plt.subplots(3,2) 
grid[0,0].axis('off')
grid[0,1].axis('off')
grid[1,0].axis('off')
grid[1,1].axis('off')
grid[2,0].axis('off')
grid[2,1].axis('off')


grid[0,0].set_title("TM_CCOEFF")
grid[0,1].set_title("Detected image")
grid[1,0].set_title("TM_SQDIFF")
grid[1,1].set_title("Detected image")
grid[2,0].set_title("TM_CCORR")
grid[2,1].set_title("Detected image")

res=cv2.matchTemplate(img,cara,cv2.TM_CCOEFF)
grid[0,0].imshow(res,cmap='gray')
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
cv2.rectangle(img,max_loc, (max_loc[0]+300,max_loc[1]+300), 255, 4)
grid[0,1].imshow(img)

res1=cv2.matchTemplate(copy,cara,cv2.TM_SQDIFF)
grid[1,0].imshow(res1,cmap='gray')
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res1)
cv2.rectangle(copy,max_loc, (min_loc[0]+300,min_loc[1]+300), 255, 4)
grid[1,1].imshow(copy)

res2=cv2.matchTemplate(copy2,cara,cv2.TM_CCORR)
grid[2,0].imshow(res2,cmap='gray')
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res2)
cv2.rectangle(copy2,max_loc, (max_loc[0]+300,max_loc[1]+300), 255, 4)
grid[2,1].imshow(copy2)