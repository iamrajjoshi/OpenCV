import cv2 as cv
import numpy as np
import imutils
import time
from imutils.video import VideoStream
import os

faceCascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
face_id = input('\n[INPUT] Enter user id: ')
print("[INFO] Initializing face capture. Look the camera and wait ...")
video_feed = VideoStream(src=1).start()
time.sleep(2.0)
count = 0
print("[INFO] Starting capture...")
while ((cv.waitKey(1) & 0xFF) != ord("q")):
    if count >= 30:
        break
    frame = video_feed.read() #get webcam feed
    frame = imutils.resize(frame, width=500)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #grayscale image
    found = faceCascade.detectMultiScale (gray, scaleFactor=1.1, minNeighbors=10, minSize = (30, 30)) #detect faces
    for (x,y,w,h) in found:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        count += 1
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv.imwrite("dataset/User." +
                    str(face_id) + '.' +
                    str(count) +
                    ".jpg", gray[y:y+h,x:x+w])
    cv.imshow('Image',frame)
    
print ("[INFO] Finished")
cv.destroyAllWindows()
