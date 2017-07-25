import logging
logging.basicConfig(level=logging.INFO)

import time
import numpy as np
import cv2
import pyrealsense as pyrs
import scipy.misc
import IPython

cap = cv2.VideoCapture('final.mov')
fgbg = cv2.createBackgroundSubtractorMOG2()

ret, frame = cap.read()

#Initialize background subtractor
for i in range(50):
    fmask = fgbg.apply(frame)

#Output functions
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter('output.mp4', fourcc, 60.0, (frame.shape[1], frame.shape[0]))

count = 0
pts = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    count += 1

    #Apply background subtractor to new frames
    fmask = fgbg.apply(frame)
    fmask = cv2.GaussianBlur(fmask, (3, 3), 0)
    fmask = cv2.erode(fmask, None, iterations=2)
    fmask = cv2.dilate(fmask, None, iterations=2)

    #Find contours
    cnts = cv2.findContours(fmask.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)[-2]

    #Find circle and draw it on center
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.circle(frame, center, 10, (0, 0, 255), -1)
        pts.append(center)

    #Draw red line from start to finish 
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 6)

    #Show images
    cv2.imshow('Mask', fmask)
    cv2.imshow('Frame', frame)

    #Write to output
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(count)



print(count)
cap.release()