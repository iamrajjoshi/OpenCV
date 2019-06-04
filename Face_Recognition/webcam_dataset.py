from dependencies import *

training_cascade_path = "haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(training_cascade_path)
video_feed = cv.VideoCapture(0)
face_id = input('[INPUT] Enter your user number: ')

video_feed.set(3, 640)
video_feed.set(4, 480)
count = 0
while True:
	trash, frame = video_feed.read() #gets webcam feed
	grayscale_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #grayscale image
	faces_found = faceCascade.detectMultiScale (grayscale_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) #detect faces in the grayscale image
	
	for (x, y, w, h) in faces_found: 
		cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)#draw a rectangle around the faces
		count += 1

		cv.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", grayscale_image[y:y+h,x:x+w])
		cv.imshow('Video Feed', frame)
	
	if chr(cv.waitKey(1) & 255) == 'q': break
	elif count >= 30: break #amount of data set pictures taken

print("[INFO] Pictures Taken, Dataset Collected")
video_feed.release()
cv.destroyAllWindows()