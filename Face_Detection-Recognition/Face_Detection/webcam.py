from imports import *

training_cascade_path = "../Haar Cascade/haarcascade_frontalface_default.xml" #change path to where it is stored
face_cascade = cv.CascadeClassifier(training_cascade_path)
video_feed = cv.VideoCapture(0)

print("[INFO] starting video stream...")
time.sleep(2.0) #taking in account the lag for the video feed to start
fps = FPS().start()

while True:
	trash, frame = video_feed.read() #gets webcam feed
	grayscale_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #grayscale image
	
	num_faces_found = face_cascade.detectMultiScale (grayscale_image, scaleFactor=1.1, minNeighbors=5, minSize = (30, 30)) #detect faces in the grayscale image
	for (x, y, w, h) in num_faces_found: cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) #draw a rectangle around the faces
	cv.imshow('Video Feed', frame)
	fps.update()
	if chr(cv.waitKey(0) & 255) == 'q' : break

video_feed.release() #turn camera off
fps.stop()
print ("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print ("[INFO] FPS: {:.2f}".format(fps.fps()))
cv.destroyAllWindows()