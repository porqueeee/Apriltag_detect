import cv2

import depthai as dai

import numpy as np

# Color definitions
colors = ["red", "yellow", "blue", "orange"]
pixel_counts = {color: 0 for color in colors}
centroids = {color: [0, 0] for color in colors}
color_ranges = {
    "red": (np.array([0, 125, 172]), np.array([69, 255, 255])),
    "yellow": (np.array([18, 60, 142]), np.array([50, 255, 255])),
    "blue": (np.array([100, 93, 46]), np.array([133, 255, 223])),
    "orange": (np.array([0, 150, 117]), np.array([66, 255, 255])),
}


# Create pipeline

with dai.Pipeline() as pipeline:

    # Define source and output

    cam = pipeline.create(dai.node.Camera).build()

    videoQueue = cam.requestOutput((640,400)).createOutputQueue()


    # Connect to device and start pipeline

    pipeline.start()

    while pipeline.isRunning():

        videoIn = videoQueue.get()

        assert isinstance(videoIn, dai.ImgFrame)

        #cv2.imshow("video", videoIn.getCvFrame())

        frame = videoIn.getCvFrame()

        frame = cv2.resize(frame, (640, 480))

        # bgr to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # restart pixel counts and centroids
        for color in colors:
            pixel_counts[color] = 0
            centroids[color] = [0, 0]

        # Detect dominant color
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
            pixel_counts[color] = cv2.countNonZero(mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    centroids[color] = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]

        dominant_color = max(pixel_counts, key=pixel_counts.get)
        if pixel_counts[dominant_color] > 0:
            cX, cY = centroids[dominant_color]
            upper_left = (max(cX - 20, 0), max(cY - 20, 0))
            bottom_right = (min(cX + 20, 639), min(cY + 20, 479))
            norm_ul = (upper_left[0] / 640, upper_left[1] / 480)
            norm_br = (bottom_right[0] / 640, bottom_right[1] / 480)
            
            

        # most dominant color by pixel count
        dominant_color = max(pixel_counts, key=pixel_counts.get)

        # show centroid
        if pixel_counts[dominant_color] > 0:
            cX, cY = centroids[dominant_color]
            cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
            cv2.putText(frame, f"{dominant_color} ({cX}, {cY})", (cX - 50, cY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        cv2.imshow("Vista de la cámara", frame)


        if cv2.waitKey(1) == ord("q"):

            break