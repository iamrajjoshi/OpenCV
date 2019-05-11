from imports import *

image_path = sys.argv[1] #get image
training_cascade_path = "../Haar Cascade/haarcascade_frontalface_default.xml" #training data path (change for where i put it)

face_cascade = cv.CascadeClassifier(training_cascade_path) #create the haarcascade

image = cv.imread(image_path)
grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #grayscale image

num_faces_found = face_cascade.detectMultiScale(grayscale_image, scaleFactor = 1.35, minNeighbors = 5, minSize = (30, 30)) #detect faces in the grayscale image
for (x, y, w, h) in num_faces_found: cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2) #draw a rectangle around the faces
print("Number of faces found: ", len(num_faces_found))
cv.imshow("Face Detection", help_func.resize(image, height = 500, width = 500))

if not os.path.exists('detected_images'): #make dir if not exists
    os.makedirs('detected_images')
new_dir = sys.argv[1].split('/')
cv.imwrite("detected_images/detected_" + str(new_dir[-1]), image) #add to folder

while chr(cv.waitKey(0) & 255) != 'q': continue
cv.destroyAllWindows()