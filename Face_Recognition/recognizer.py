import cv2 as cv
import numpy as np
import os
import imutils
import time
from imutils.video import VideoStream
from imutils.video import FPS

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascadePath);
font = cv.FONT_HERSHEY_SIMPLEX

id = 0

file = open("names.txt", "r")
names = (file.read()).split()
video_feed = VideoStream(src=1).start()
print("[INFO] starting video stream...")
time.sleep(2.0)
fps = FPS().start()

while ((cv.waitKey(1) & 0xFF) != ord("q")):
    frame = video_feed.read() #get webcam feed
    frame = imutils.resize(frame, width=500)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #grayscale image
    found = faceCascade.detectMultiScale (gray, scaleFactor=1.1, minNeighbors=10, minSize = (30, 30)) #detect faces

    for(x,y,w,h) in found:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  

    cv.imshow('Video Feed',frame)
    fps.update()
        
fps.stop()
print ("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print ("[INFO] FPS: {:.2f}".format(fps.fps()))
cv.destroyAllWindows()
