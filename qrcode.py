import cv2
import numpy as np

img = cv2.imread("qrcode-feature.jpg")
#img = cv2.imread("apriltagtest.png")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
detector = cv2.QRCodeDetector()
data, bbox, straight_qrcode = detector.detectAndDecode(gray)
print(data)
# if there is a QR code
if bbox is not None:
	#print(f"QRCode data:\n{data}")
	# display the image with lines
	# length of bounding box
	n_lines = len(bbox)
	for i in range(n_lines):
		# draw all lines
		point1 = tuple(bbox[i][0])
		point2 = tuple(bbox[(i+1) % n_lines][0])
		cv2.polylines(img, np.int32([bbox]), True, (255, 0, 0), 2)

cv2.imshow("Detection", img)
cv2.waitKey(0)
