import cv2 as cv
import numpy as np
import imutils
import time
from imutils.video import VideoStream
from imutils.video import FPS
import os
from PIL import Image

new_face = 0
while True:
    #dataset.py
    faceCascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    print("\n[INFO] Entering Capture Stage...")
    i = input ('[INPUT] Would you like to add a person? {y or n}: ')
    if i == "n":
        break
    new_face+=1
    face_id = input('[INPUT] Enter user id: ')
    print("[INFO] Initializing face capture. Look the camera and wait...")
    video_feed = VideoStream(src=0).start()
    time.sleep(2.0)
    count = 0
    print("[INFO] Starting capture...")
    while ((cv.waitKey(1) & 0xFF) != ord("q")):
        if count >= 30:
            break
        frame = video_feed.read() #get webcam feed
        frame = cv.flip(frame, -1)
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
    video_feed.stream.release()
    cv.destroyAllWindows()

#trainer
if new_face > 0:
    path = 'dataset'
    recognizer = cv.face.LBPHFaceRecognizer_create()
    detector = cv.CascadeClassifier("haarcascade_frontalface_default.xml");
    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        return faceSamples,ids
    print("\n[INFO] Entering Training Stage...")
    print ("[INFO] Training faces. It will take a few seconds. Wait...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml')
    print("[INFO] {0} faces trained ...".format(len(np.unique(ids))))


#recognizer
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascadePath);
font = cv.FONT_HERSHEY_SIMPLEX

id = 0

file = open("names.txt", "r")
names = (file.read()).split()
video_feed = VideoStream(src=0).start()
print("\n[INFO] Entering Stream Stage...")
print("[INFO] starting video stream...")
time.sleep(2.0)
fps = FPS().start()

while ((cv.waitKey(1) & 0xFF) != ord("q")):
    frame = video_feed.read() #get webcam feed
    frame = cv.flip(frame, -1)
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
