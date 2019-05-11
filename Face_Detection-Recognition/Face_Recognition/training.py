from dependencies import *

path = 'dataset' #path of dataset
training_cascade_path = cv.CascadeClassifier("haarcascade_frontalface_default.xml");

#get the images and label data
def getImagesAndLabels(path):

	imagePaths = [os.path.join(path,f) for f in os.listdir(path)]	 
	faceSamples=[]
	user_number = []
	
	for imagePath in imagePaths:
		PIL_img = Image.open(imagePath).convert('L') #convert to grayscale
		PIL_img = Image.open(imagePath)
		img_numpy = np.array(PIL_img,'uint8')
	
		user_number = int(os.path.split(imagePath)[-1].split(".")[1])
		faces = training_cascade_path.detectMultiScale(img_numpy)

		for (x,y,w,h) in faces:
			faceSamples.append(img_numpy[y:y+h,x:x+w])
			user_number.append(user_number)
	return faceSamples,user_number

faces,user_number = getImagesAndLabels(path)
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(user_number))
recognizer.write('trainer/trainer.yml') #save the model into trainer/trainer.yml

print("[INFO] Training Completed. Faces Trained: ", len(np.unique(user_number)))