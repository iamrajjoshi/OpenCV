from dependencies import *

path = 'input_images' #path of dataset
training_cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv.CascadeClassifier(training_cascade_path)
user_number = input ('[INPUT] Enter your user number: ')
picture_number = 0 #picture number ---> change to what you want it to be
image_paths = [os.path.join(path,f) for f in os.listdir(path)] #get path of all images in the directory
for image_path in image_paths:
	r = cv.imread(image_path)
	#r = cv.flip( r, 1 ) # flip the image for better regonition
	grayscale_image = cv.cvtColor(r, cv.COLOR_BGR2GRAY) #grayscale image
	cv.imwrite("dataset/User." + str(user_number) + '.' + str(picture_number) + ".jpg", grayscale_image) #save to output dataset directory directory
	picture_number += 1
	
print("[INFO] Pictures Taken, Dataset Collected")