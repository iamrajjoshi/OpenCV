from dependencies import *

recognizer = cv.face.LBPHFaceRecognizer_create() #local binary patterns historgrams
recognizer.read('trainer/trainer.yml') #path where I stored yml file
training_cascade_path = "../Haar Cascade/haarcascade_frontalface_default.xml" #path to harrcascade
face_cascade = cv.CascadeClassifier(training_cascade_path);
font = cv.FONT_HERSHEY_COMPLEX

user_number = 0 #user_number counter
names = ['None', 'Raj', 'Rakesh', 'Jigisha'] #hard coded to the user numbers in the dataset stage
print("[INFO] starting video stream...")
time.sleep(2.0) #take in account lag to start camera
fps = FPS().start()
video_feed = cv.VideoCapture(0)
video_feed.set(3, 640)
video_feed.set(4, 480)

minW = 0.1*video_feed.get(3)
minH = 0.1*video_feed.get(4)
while True:
	trash, frame = video_feed.read() #gets webcam feed
	grayscale_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #grayscale image
	
	num_faces_found = face_cascade.detectMultiScale (grayscale_image, scaleFactor=1.1, minNeighbors=5, minSize = (30, 30)) #detect faces in the grayscale image

	for(x,y,w,h) in num_faces_found:
		cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
		user_number, confidence = recognizer.predict(grayscale_image[y:y+h,x:x+w]) #confidence of 0 is perfect
		
		if (confidence < 100):
			user_number = names[user_number]
			confidence = "  {0}%".format(round(100 - confidence))
		else: #if you find a face, but it is not in detected as known
			user_number = "unknown"
			confidence = "  {0}%".format(round(100 - confidence))
		
		cv.putText(frame, str(user_number), (x+5,y-5), font, 1, (255,255,255), 2)
		cv.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
	fps.update()
	cv.imshow('Camera',frame) 
	if chr(cv.waitKey(1) & 255) == 'q': break
video_feed.release()
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] FPS: {:.2f}".format(fps.fps()))
cv.destroyAllWindows()