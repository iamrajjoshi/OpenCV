from imports import *

image = cv.imread(sys.argv[1])
ratio = image.shape[0] / 500.0 #load the image and compute the ratio of the old height to 500 pixels (used for convience) - ratio used for four point transformation
small_image = image.copy() #clone the original picture, and resize it to 500 pixels
image = help_func.resize(image, height = 500)

grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #grayscale
grayscale = cv.GaussianBlur(grayscale, (5, 5), 0) #blur for next part
edged = cv.Canny(grayscale, 75, 200) #find edges
'''
cv.imshow("Original", image)
cv.imshow("Edge Detection", edged)
cv.waitKey()
cv.destroyAllWindows()
'''
contours = cv.findContours(edged, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) #find contours of edged image
contours = contours[1]
contours = sorted(contours, key = cv.contourArea, reverse = True)[:5] # Largest, so that if there is more distinction in the image, it still picks up the right thing
for index in contours:
	perimeter = cv.arcLength(index, True)
	estimate = cv.approxPolyDP(index, 0.02 * perimeter, True) #estimate contour
	if len(estimate) == 4: #if our approximated contour has four point then the document has been found
		screenCnt = estimate
		break
'''
cv.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv.imshow("Outline", image)
cv.waitKey()
cv.destroyAllWindows()

'''
birds_eye = fpt (small_image, screenCnt.reshape(4, 2) * ratio) #apply the four point transformation to get brids-eye view of the original image multiply cause using smaller image, so have to use ratio
birds_eye = cv.cvtColor(birds_eye, cv.COLOR_BGR2GRAY)
threshold = ts(birds_eye, 11, offset = 10, method = "gaussian")
birds_eye = (birds_eye > threshold).astype("uint8") * 255 #convert the birds_eye image to grayscale, then threshold it to give it that 'black and white' paper effect

cv.imshow("Original", help_func.resize(small_image, height = 700))
cv.imshow("Scanned", help_func.resize(birds_eye, height = 700))

if not os.path.exists('scanned_images'): #make dir if not exists
    os.makedirs('scanned_images')
new_dir = sys.argv[1].split('/')
cv.imwrite("scanned_images/scanned_" + str(new_dir[-1]), birds_eye) #add to folder
cv.waitKey()